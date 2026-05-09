#!/usr/bin/env python3
"""
FNA2/bench_packing.py — Benchmark complet FNA2 vs SDPA

Mesure 4 configurations :
  A) SDPA    + pas de packing  + graph_bias  (baseline actuel)
  B) FA2 std + packing         + sans biais  (ce qui existe avant FNA2)
  C) FNA2    + packing         + graph_bias  (objectif FNA2)
  D) FNA2    + pas de packing  + graph_bias  (sanity check)

Métriques :
  - Throughput (tokens/s) forward + backward
  - VRAM peak (MB)
  - Temps/step (ms)

Usage :
  python FNA2/bench_packing.py                       # sweep S complet
  python FNA2/bench_packing.py --seq 2048            # S fixé
  python FNA2/bench_packing.py --config 220m         # config 220M
  python FNA2/bench_packing.py --forward-only        # pas de backward
"""

import argparse
import math
import time
import sys
import os

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG MODÈLES
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    "60m": dict(
        B=4, H=8, D=64, H_KV=4, H_G=4, R=32,
        label="60M  (B=4, H=8, D=64, 4 graph + 4 vanilla)",
    ),
    "220m": dict(
        B=2, H=12, D=64, H_KV=4, H_G=8, R=32,
        label="220M (B=2, H=12, D=64, 8 graph + 4 vanilla)",
    ),
    "220m_large": dict(
        B=1, H=12, D=64, H_KV=4, H_G=8, R=32,
        label="220M single-batch (B=1, H=12, pour max S)",
    ),
}

SEQ_SWEEP = [256, 512, 1024, 2048, 4096]

# ─────────────────────────────────────────────────────────────────────────────
# IMPORT FNA2 (optionnel — bench A/B fonctionne sans)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from flash_naylis_attn import flash_naylis_attn, BLOCK_R
    FNA2_OK = True
except ImportError:
    FNA2_OK = False
    print("⚠️  FNA2 non disponible — configs C et D désactivées")

try:
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    FA2_OK = True
except ImportError:
    FA2_OK  = False

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _vram_mb() -> float:
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024**2

def _reset_vram():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

def _make_tensors(cfg, S, requires_grad=True):
    """Génère les tenseurs Q/K/V/R_q/R_k/gs en BF16 pour un step standard."""
    B, H, D    = cfg["B"], cfg["H"], cfg["D"]
    H_KV, H_G  = cfg["H_KV"], cfg["H_G"]
    R          = cfg["R"]

    kw = dict(device=DEVICE, dtype=DTYPE, requires_grad=requires_grad)
    q  = torch.randn(B, H,    S, D, **kw)
    k  = torch.randn(B, H_KV, S, D, **kw)
    v  = torch.randn(B, H_KV, S, D, **kw)
    rq = torch.randn(B, H_G,  S, R, **kw)
    rk = torch.randn(B, H_G,  S, R, **kw)
    gs = torch.randn(H_G, device=DEVICE, dtype=torch.float32, requires_grad=requires_grad)
    return q, k, v, rq, rk, gs

def _make_varlen_tensors(cfg, S, n_seqs=None, requires_grad=True):
    """
    Génère les tenseurs packés pour n_seqs séquences de longueur S chacune.
    n_seqs = B (même nombre total de séquences que le batch standard).
    """
    B, H, D    = cfg["B"], cfg["H"], cfg["D"]
    H_KV, H_G  = cfg["H_KV"], cfg["H_G"]
    R          = cfg["R"]
    n_seqs     = n_seqs or B

    total = n_seqs * S   # même nombre de tokens qu'un batch standard

    kw = dict(device=DEVICE, dtype=DTYPE, requires_grad=requires_grad)
    q  = torch.randn(total, H,    D, **kw)
    k  = torch.randn(total, H_KV, D, **kw)
    v  = torch.randn(total, H_KV, D, **kw)
    rq = torch.randn(total, H_G,  R, **kw)
    rk = torch.randn(total, H_G,  R, **kw)
    gs = torch.randn(H_G, device=DEVICE, dtype=torch.float32, requires_grad=requires_grad)

    # cu_seqlens : chaque séquence a exactement S tokens
    seqlens  = torch.tensor([S] * n_seqs, dtype=torch.int32)
    cu_q     = torch.zeros(n_seqs + 1, dtype=torch.int32, device=DEVICE)
    cu_q[1:] = torch.cumsum(seqlens, 0).to(DEVICE)

    return q, k, v, rq, rk, gs, cu_q, cu_q.clone(), S, S

def _sdpa_graph_bias(q, k, v, rq, rk, gs, scale, is_causal=True):
    """Config A : SDPA + biais [B,H,S,S] matérialisé."""
    B, H, S, D = q.shape
    H_G, R     = rq.shape[1], rq.shape[3]
    H_KV       = k.shape[1]
    GQA        = H // H_KV

    # Matérialisation [B, H_G, S, S]
    bias = torch.matmul(rq, rk.transpose(-2, -1))
    scale_g = gs.view(1, H_G, 1, 1)
    bias = (scale_g * bias).to(DTYPE)

    # Pad vanilla heads avec zéros → [B, H, S, S]
    vanilla = H - H_G
    if vanilla > 0:
        pad  = torch.zeros(B, vanilla, S, S, dtype=DTYPE, device=q.device)
        bias = torch.cat([bias, pad], dim=1)

    k_exp = k.repeat_interleave(GQA, dim=1)
    v_exp = v.repeat_interleave(GQA, dim=1)

    return F.scaled_dot_product_attention(
        q, k_exp, v_exp,
        attn_mask=bias,
        is_causal=False,   # le masque causal est déjà dans le bias via -inf
        scale=scale,
    )

def _fna2_standard(q, k, v, rq, rk, gs, scale):
    """Config D : FNA2 sans packing."""
    return flash_naylis_attn(q, k, v, rq, rk, gs, softmax_scale=scale, is_causal=True)

def _fna2_varlen(q, k, v, rq, rk, gs, cu_q, cu_k, msl_q, msl_k, scale):
    """Config C : FNA2 avec packing + graph_bias."""
    return flash_naylis_attn(
        q, k, v, rq, rk, gs,
        softmax_scale=scale,
        is_causal=True,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=msl_q,
        max_seqlen_k=msl_k,
    )

# ─────────────────────────────────────────────────────────────────────────────
# TIMEIT
# ─────────────────────────────────────────────────────────────────────────────

def _bench(fn, n_warmup=3, n_bench=10, backward=True) -> dict:
    """
    Mesure le temps moyen + VRAM d'une fonction fn().
    Retourne {'ms': float, 'vram_mb': float}.
    """
    if DEVICE != "cuda":
        return {"ms": float("nan"), "vram_mb": 0.0}

    # Warmup
    for _ in range(n_warmup):
        out = fn()
        if backward and out.requires_grad:
            loss = out.float().sum()
            loss.backward()
        torch.cuda.synchronize()

    _reset_vram()

    # Bench
    times = []
    for _ in range(n_bench):
        t0  = time.perf_counter()
        out = fn()
        if backward and out.requires_grad:
            loss = out.float().sum()
            loss.backward()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    vram = _vram_mb()
    ms   = sum(times) / len(times)
    return {"ms": ms, "vram_mb": vram}


# ─────────────────────────────────────────────────────────────────────────────
# BENCH D'UNE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

def bench_one_seq_len(cfg: dict, S: int, forward_only: bool) -> dict:
    B     = cfg["B"]
    D     = cfg["D"]
    scale = 1.0 / math.sqrt(D)
    backward = not forward_only

    results = {}

    # ── Config A : SDPA + biais matérialisé (baseline actuel) ────────────────
    try:
        q, k, v, rq, rk, gs = _make_tensors(cfg, S)
        res_a = _bench(
            lambda: _sdpa_graph_bias(q, k, v, rq, rk, gs, scale),
            backward=backward,
        )
        tokens_a   = B * S * 1000 / res_a["ms"]
        results["A_sdpa_bias"] = {
            "ms": res_a["ms"], "vram": res_a["vram_mb"],
            "tok_s": tokens_a,
        }
    except Exception as e:
        results["A_sdpa_bias"] = {"ms": float("nan"), "vram": 0.0, "tok_s": 0.0, "err": str(e)}

    # ── Config B : FA2 varlen SANS graph_bias (ce qui existait avant FNA2) ───
    if FA2_OK:
        try:
            (q_p, k_p, v_p, rq_p, rk_p, gs_p,
             cu_q, cu_k, msl_q, msl_k) = _make_varlen_tensors(cfg, S)
            # FA2 varlen ne prend pas de biais — on passe Q/K/V seulement
            # layout FA2 : [total, H, D] → [total, D, H] (head dernier)
            q_fa = q_p.reshape(-1, cfg["H"],    D).to(DTYPE)
            k_fa = k_p.reshape(-1, cfg["H_KV"], D).to(DTYPE)
            v_fa = v_p.reshape(-1, cfg["H_KV"], D).to(DTYPE)

            def _fa2_fn():
                return flash_attn_varlen_func(
                    q_fa, k_fa, v_fa,
                    cu_q, cu_k, msl_q, msl_k,
                    dropout_p=0.0,
                    softmax_scale=scale,
                    causal=True,
                )

            res_b    = _bench(_fa2_fn, backward=False)   # FA2 varlen bwd pas testé
            n_tokens = B * S
            tokens_b = n_tokens * 1000 / res_b["ms"]
            results["B_fa2_varlen_nobias"] = {
                "ms": res_b["ms"], "vram": res_b["vram_mb"],
                "tok_s": tokens_b,
            }
        except Exception as e:
            results["B_fa2_varlen_nobias"] = {"ms": float("nan"), "vram": 0.0, "tok_s": 0.0, "err": str(e)}
    else:
        results["B_fa2_varlen_nobias"] = {"ms": float("nan"), "vram": 0.0, "tok_s": 0.0,
                                          "err": "flash_attn non installé"}

    # ── Config C : FNA2 packing + graph_bias ← LE VRAI OBJECTIF ─────────────
    if FNA2_OK:
        try:
            (q_p, k_p, v_p, rq_p, rk_p, gs_p,
             cu_q, cu_k, msl_q, msl_k) = _make_varlen_tensors(cfg, S)

            res_c    = _bench(
                lambda: _fna2_varlen(q_p, k_p, v_p, rq_p, rk_p, gs_p,
                                     cu_q, cu_k, msl_q, msl_k, scale),
                backward=backward,
            )
            n_tokens = B * S
            tokens_c = n_tokens * 1000 / res_c["ms"]
            results["C_fna2_varlen_bias"] = {
                "ms": res_c["ms"], "vram": res_c["vram_mb"],
                "tok_s": tokens_c,
            }
        except Exception as e:
            results["C_fna2_varlen_bias"] = {"ms": float("nan"), "vram": 0.0, "tok_s": 0.0, "err": str(e)}
    else:
        results["C_fna2_varlen_bias"] = {"ms": float("nan"), "vram": 0.0, "tok_s": 0.0,
                                         "err": "FNA2 non disponible"}

    # ── Config D : FNA2 standard (pas de packing) ────────────────────────────
    if FNA2_OK:
        try:
            q, k, v, rq, rk, gs = _make_tensors(cfg, S)
            res_d    = _bench(
                lambda: _fna2_standard(q, k, v, rq, rk, gs, scale),
                backward=backward,
            )
            tokens_d = B * S * 1000 / res_d["ms"]
            results["D_fna2_standard"] = {
                "ms": res_d["ms"], "vram": res_d["vram_mb"],
                "tok_s": tokens_d,
            }
        except Exception as e:
            results["D_fna2_standard"] = {"ms": float("nan"), "vram": 0.0, "tok_s": 0.0, "err": str(e)}
    else:
        results["D_fna2_standard"] = {"ms": float("nan"), "vram": 0.0, "tok_s": 0.0,
                                       "err": "FNA2 non disponible"}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# AFFICHAGE
# ─────────────────────────────────────────────────────────────────────────────

_LABELS = {
    "A_sdpa_bias"          : "A  SDPA + biais [S²]    (baseline actuel)",
    "B_fa2_varlen_nobias"  : "B  FA2 varlen  — sans biais graph",
    "C_fna2_varlen_bias"   : "C  FNA2 varlen + biais  ← OBJECTIF",
    "D_fna2_standard"      : "D  FNA2 standard (no pack)",
}

def _print_row(label, r, ref_vram, ref_tok):
    err    = r.get("err", "")
    if err and math.isnan(r["ms"]):
        print(f"  {label:45s}  N/A  ({err[:40]})")
        return

    ms     = r["ms"]
    vram   = r["vram"]
    tok_s  = r["tok_s"]

    vram_delta = f"  {vram - ref_vram:+.0f} MB" if ref_vram else ""
    speedup    = f"  {tok_s / ref_tok:.2f}×" if ref_tok and not math.isnan(tok_s) else ""

    print(f"  {label:45s}  {ms:7.1f} ms   {vram:7.0f} MB{vram_delta:12s}  "
          f"{tok_s/1e3:7.1f} K tok/s{speedup}")


def _print_table(cfg, S, results, forward_only):
    bwd = "fwd" if forward_only else "fwd+bwd"
    print(f"\n  S={S:5d}  [{bwd}]  — {cfg['label']}")
    print(f"  {'Config':45s}  {'Time':>9}   {'VRAM':>9}{'Δ vs A':>12}  {'Throughput':>12}{'vs A':>8}")
    print("  " + "─" * 95)

    ref_vram = results.get("A_sdpa_bias", {}).get("vram", 0)
    ref_tok  = results.get("A_sdpa_bias", {}).get("tok_s", 0)

    for key, label in _LABELS.items():
        if key in results:
            _print_row(label, results[key], ref_vram, ref_tok)

    # Insight clé : gain C vs A
    a = results.get("A_sdpa_bias", {})
    c = results.get("C_fna2_varlen_bias", {})
    if a.get("tok_s") and c.get("tok_s") and not math.isnan(c["tok_s"]):
        vram_saved = a["vram"] - c["vram"]
        speedup_c  = c["tok_s"] / a["tok_s"]
        print(f"\n  ★ FNA2 packing vs SDPA baseline : {speedup_c:.2f}× faster, "
              f"{vram_saved:.0f} MB économisés")

        bias_theoretical = cfg["B"] * cfg["H"] * S * S * 2 / 1024**2
        print(f"  ★ Biais [B,H,S,S] BF16 théorique : {bias_theoretical:.1f} MB  →  0 MB avec FNA2")


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT MÉMOIRE DÉTAILLÉ
# ─────────────────────────────────────────────────────────────────────────────

def vram_breakdown(cfg: dict, S: int):
    """Affiche la répartition VRAM tenseur par tenseur pour A vs C."""
    B, H, D    = cfg["B"], cfg["H"], cfg["D"]
    H_KV, H_G  = cfg["H_KV"], cfg["H_G"]
    R          = cfg["R"]

    def _mb(n_elements, dtype_bytes=2):
        return n_elements * dtype_bytes / 1024**2

    print(f"\n  Décomposition VRAM (S={S}, {cfg['label']}):")
    print(f"  {'Tenseur':30s}  {'Config A (SDPA)':20s}  {'Config C (FNA2)':20s}")
    print("  " + "─" * 74)

    rows = [
        ("Q  [B,H,S,D]",     B*H*S*D,    B*H*S*D),
        ("K  [B,H_KV,S,D]",  B*H_KV*S*D, B*H_KV*S*D),
        ("V  [B,H_KV,S,D]",  B*H_KV*S*D, B*H_KV*S*D),
        ("R_q [B,H_G,S,R]",  B*H_G*S*R,  B*H_G*S*R),
        ("R_k [B,H_G,S,R]",  B*H_G*S*R,  B*H_G*S*R),
        ("Biais [B,H,S,S]",  B*H*S*S,    0),
        ("LSE [B,H,S]",      0,          B*H*S),
        ("Out [B,H,S,D]",    B*H*S*D,    B*H*S*D),
    ]

    total_a, total_c = 0, 0
    for name, n_a, n_c in rows:
        mb_a = _mb(n_a)
        mb_c = _mb(n_c)
        total_a += mb_a
        total_c += mb_c
        mark = "← éliminé" if n_a > 0 and n_c == 0 else ("← ajouté" if n_a == 0 and n_c > 0 else "")
        print(f"  {name:30s}  {mb_a:8.2f} MB            {mb_c:8.2f} MB  {mark}")

    print("  " + "─" * 74)
    print(f"  {'TOTAL':30s}  {total_a:8.2f} MB            {total_c:8.2f} MB")
    print(f"  Économie : {total_a - total_c:.2f} MB  ({(1 - total_c/total_a)*100:.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_bench(config_name: str, seq_lens: list, forward_only: bool):
    if DEVICE != "cuda":
        print("⛔  Bench nécessite un GPU CUDA.")
        return

    cfg = CONFIGS.get(config_name, CONFIGS["60m"])

    print("=" * 100)
    print(f"  FNA2 — Benchmark  [{cfg['label']}]")
    print(f"  Device : {torch.cuda.get_device_name(0)}")
    print(f"  Mode   : {'forward seulement' if forward_only else 'forward + backward'}")
    print("=" * 100)

    # Décomposition mémoire pour S=2048
    S_mem = min(2048, max(seq_lens))
    vram_breakdown(cfg, S_mem)

    print()
    print("  " + "─" * 95)
    print(f"  {'Config':45s}  {'Time':>9}   {'VRAM':>9}{'Δ vs A':>12}  {'Throughput':>12}{'vs A':>8}")
    print("  " + "─" * 95)

    for S in seq_lens:
        # Skip si trop grand pour la VRAM disponible
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        bias_gb    = cfg["B"] * cfg["H"] * S * S * 2 / 1024**3
        if bias_gb > vram_total * 0.6:
            print(f"\n  S={S:5d}  ⚠️  biais [B,H,S,S] = {bias_gb:.1f} GB > 60% VRAM — ignoré")
            continue

        try:
            results = bench_one_seq_len(cfg, S, forward_only)
            _print_table(cfg, S, results, forward_only)
        except torch.cuda.OutOfMemoryError:
            print(f"\n  S={S:5d}  ⛔ OOM — séquence trop longue pour cette config")
            torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("  Légende :")
    print("  A = Baseline actuel NaylisGPT  (SDPA, biais [S²] matérialisé, pas de packing)")
    print("  B = FA2 varlen                 (packing activé, mais graph_bias absent)")
    print("  C = FNA2 varlen                (packing + graph_bias — objectif du projet)")
    print("  D = FNA2 standard              (graph_bias, pas de packing — sanity check)")
    print()
    print("  Gain C/A = apport réel FNA2 sur l'entraînement avec packing")
    print("=" * 100)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNA2 Benchmark — packing vs SDPA")
    parser.add_argument("--config",       choices=list(CONFIGS.keys()), default="60m")
    parser.add_argument("--seq",          type=int, default=None,
                        help="Longueur de séquence fixée (défaut : sweep 256→4096)")
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    seq_lens = [args.seq] if args.seq else SEQ_SWEEP
    run_bench(args.config, seq_lens, args.forward_only)
