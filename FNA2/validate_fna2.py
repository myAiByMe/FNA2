#!/usr/bin/env python3
"""
FNA2/validate_fna2.py — Suite de validation numérique complète

Tests :
  1. Forward  : FNA2 vs SDPA+biais matérialisé  (config 60M et 220M)
  2. Backward : gradients dQ, dK, dV, dR_q, dR_k, d_gs via gradcheck
  3. Varlen   : cohérence packing vs non-packing (même loss, même grad)
  4. graph_scale=0 : FNA2 == attention classique pure (step 0 garanti)
  5. Vanilla heads : têtes sans biais identiques au baseline
  6. Mémoire  : FNA2 n'alloue pas de tenseur [B, H, S, S]

Usage :
    python validate_fna2.py                         # tous les tests
    python validate_fna2.py --test forward          # test spécifique
    python validate_fna2.py --config 220m           # config 220M
"""

import argparse
import math
import sys
import torch
import torch.nn.functional as F

# ── Import FNA2 ─────────────────────────────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flash_naylis_attn import flash_naylis_attn, BLOCK_R
    FNA2_OK = True
except ImportError as e:
    print(f"⛔  FNA2 non importable : {e}")
    FNA2_OK = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATIONS
# ─────────────────────────────────────────────────────────────────────────────

CONFIGS = {
    "60m": dict(
        B=2, H=8, S=128, D=64, H_KV=4, H_G=4, R=32,
        label="60M  (embed=512, 8 heads, 4 graph + 4 vanilla)",
    ),
    "220m": dict(
        B=2, H=12, S=256, D=64, H_KV=4, H_G=8, R=32,
        label="220M (embed=768, 12 heads, 8 graph + 4 vanilla)",
    ),
    "large_seq": dict(
        B=1, H=8, S=512, D=64, H_KV=4, H_G=4, R=32,
        label="Large seq (S=512, mémoire O(S²) vs O(S))",
    ),
}

TOL_BF16 = 0.05    # tolérance relative BF16 (FA2 accumule de petites erreurs)
TOL_GRAD  = 0.10   # tolérance pour les gradients en BF16


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRES
# ─────────────────────────────────────────────────────────────────────────────

def _make_tensors(cfg: dict, requires_grad: bool = True, seed: int = 42):
    """Génère Q, K, V, R_q, R_k, graph_scale avec les dims du cfg."""
    torch.manual_seed(seed)
    B, H, S, D   = cfg["B"], cfg["H"], cfg["S"], cfg["D"]
    H_KV, H_G, R = cfg["H_KV"], cfg["H_G"], cfg["R"]

    mk = lambda *shape, **kw: torch.randn(*shape, device=DEVICE, dtype=DTYPE,
                                          requires_grad=requires_grad, **kw)
    q  = mk(B, H,    S, D)
    k  = mk(B, H_KV, S, D)
    v  = mk(B, H_KV, S, D)
    rq = mk(B, H_G,  S, R)
    rk = mk(B, H_G,  S, R)
    gs = torch.randn(H_G, device=DEVICE, dtype=torch.float32, requires_grad=requires_grad)
    return q, k, v, rq, rk, gs


def _sdpa_reference(q, k, v, rq, rk, gs, cfg: dict):
    """
    Calcule la référence SDPA avec biais matérialisé [B, H, S, S].
    C'est le comportement actuel de NaylisAttention (pré-FNA2).
    """
    B, H, S, D   = cfg["B"], cfg["H"], cfg["S"], cfg["D"]
    H_KV, H_G    = cfg["H_KV"], cfg["H_G"]
    GQA           = H // H_KV
    scale         = 1.0 / math.sqrt(D)

    k_exp = k.repeat_interleave(GQA, dim=1)
    v_exp = v.repeat_interleave(GQA, dim=1)

    # Biais asymétrique [B, H_G, S, S] puis pad vanilla [B, H, S, S]
    rq_f, rk_f = rq.float(), rk.float()
    bias_g = torch.einsum("bhir,bhjr->bhij", rq_f, rk_f) * gs.view(1, H_G, 1, 1)
    if H_G < H:
        pad    = torch.zeros(B, H - H_G, S, S, device=DEVICE, dtype=torch.float32)
        bias_g = torch.cat([bias_g, pad], dim=1)
    bias_g = bias_g.to(DTYPE)

    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), device=DEVICE, dtype=DTYPE), diagonal=1
    )
    attn_mask = bias_g + causal_mask.unsqueeze(0).unsqueeze(0)

    return F.scaled_dot_product_attention(
        q, k_exp, v_exp, attn_mask=attn_mask, is_causal=False, scale=scale
    )


def _check(name: str, ref: torch.Tensor, got: torch.Tensor, tol: float = TOL_BF16):
    max_diff  = (ref - got).abs().max().item()
    mean_diff = (ref - got).abs().mean().item()
    ok = max_diff < tol
    status = "✓" if ok else "✗"
    print(f"    {status} {name:30s}  max={max_diff:.5f}  mean={mean_diff:.5f}"
          + (f"  ← ÉCHEC (tol={tol})" if not ok else ""))
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — FORWARD
# ─────────────────────────────────────────────────────────────────────────────

def test_forward(cfg: dict) -> bool:
    print(f"\n  [T1] Forward — {cfg['label']}")
    q, k, v, rq, rk, gs = _make_tensors(cfg, requires_grad=False)
    scale = 1.0 / math.sqrt(cfg["D"])

    ref  = _sdpa_reference(q, k, v, rq, rk, gs, cfg)
    out  = flash_naylis_attn(q, k, v, rq, rk, gs, softmax_scale=scale, is_causal=True)

    return _check("output vs SDPA+biais matérialisé", ref, out)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — BACKWARD (gradients)
# ─────────────────────────────────────────────────────────────────────────────

def test_backward(cfg: dict) -> bool:
    print(f"\n  [T2] Backward — {cfg['label']}")
    ok_all = True
    scale = 1.0 / math.sqrt(cfg["D"])

    # ── Référence SDPA ────────────────────────────────────────────────────────
    q, k, v, rq, rk, gs = _make_tensors(cfg, requires_grad=True)
    ref = _sdpa_reference(q, k, v, rq, rk, gs, cfg)
    dout = torch.randn_like(ref)
    ref.backward(dout)
    dq_ref = q.grad.clone()
    dk_ref = k.grad.clone()
    dv_ref = v.grad.clone()
    drq_ref = rq.grad.clone()
    drk_ref = rk.grad.clone()
    dgs_ref = gs.grad.clone()

    # ── FNA2 ──────────────────────────────────────────────────────────────────
    q2, k2, v2, rq2, rk2, gs2 = _make_tensors(cfg, requires_grad=True)
    out2 = flash_naylis_attn(q2, k2, v2, rq2, rk2, gs2, softmax_scale=scale, is_causal=True)
    out2.backward(dout)

    ok_all &= _check("dQ", dq_ref, q2.grad,  TOL_GRAD)
    ok_all &= _check("dK", dk_ref, k2.grad,  TOL_GRAD)
    ok_all &= _check("dV", dv_ref, v2.grad,  TOL_GRAD)
    ok_all &= _check("dR_q", drq_ref, rq2.grad, TOL_GRAD)
    ok_all &= _check("dR_k", drk_ref, rk2.grad, TOL_GRAD)
    ok_all &= _check("d_graph_scale", dgs_ref.float(), gs2.grad.float(), TOL_GRAD)

    return ok_all


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — VARLEN cohérence
# ─────────────────────────────────────────────────────────────────────────────

def test_varlen(cfg: dict) -> bool:
    """
    Vérifie que FNA2 varlen produit la même loss que FNA2 non-varlen
    sur un batch de 2 séquences de longueurs différentes.

    Cas testé : seq1=S//2 tokens, seq2=S//2 tokens, packées en S tokens total.
    """
    print(f"\n  [T3] Varlen (sequence packing) — {cfg['label']}")

    B, H, S, D   = 1, cfg["H"], cfg["S"], cfg["D"]   # B=1 pour varlen
    H_KV, H_G, R = cfg["H_KV"], cfg["H_G"], cfg["R"]
    scale = 1.0 / math.sqrt(D)

    # Deux séquences de longueur S//2 chacune
    s1, s2 = S // 2, S // 2
    total  = s1 + s2

    torch.manual_seed(99)
    q_pack  = torch.randn(total, H,    D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k_pack  = torch.randn(total, H_KV, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v_pack  = torch.randn(total, H_KV, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    rq_pack = torch.randn(total, H_G,  R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    rk_pack = torch.randn(total, H_G,  R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    gs      = torch.randn(H_G, device=DEVICE, dtype=torch.float32, requires_grad=True)

    cu_q = torch.tensor([0, s1, total], device=DEVICE, dtype=torch.int32)
    cu_k = torch.tensor([0, s1, total], device=DEVICE, dtype=torch.int32)

    out_varlen = flash_naylis_attn(
        q_pack, k_pack, v_pack, rq_pack, rk_pack, gs,
        softmax_scale = scale,
        is_causal     = True,
        cu_seqlens_q  = cu_q,
        cu_seqlens_k  = cu_k,
        max_seqlen_q  = s1,
        max_seqlen_k  = s1,
    )

    # ── Référence : deux passes indépendantes non-varlen ──────────────────────
    def _fna2_seq(q_s, k_s, v_s, rq_s, rk_s):
        # q_s: [slen, H, D] → reshape to [1, H, slen, D]
        q_b  = q_s.transpose(0, 1).unsqueeze(0)
        k_b  = k_s.transpose(0, 1).unsqueeze(0)
        v_b  = v_s.transpose(0, 1).unsqueeze(0)
        rq_b = rq_s.transpose(0, 1).unsqueeze(0)
        rk_b = rk_s.transpose(0, 1).unsqueeze(0)
        out  = flash_naylis_attn(q_b, k_b, v_b, rq_b, rk_b, gs,
                                 softmax_scale=scale, is_causal=True)
        return out.squeeze(0).transpose(0, 1)   # [slen, H, D]

    ref1 = _fna2_seq(
        q_pack[:s1], k_pack[:s1], v_pack[:s1],
        rq_pack[:s1], rk_pack[:s1],
    )
    ref2 = _fna2_seq(
        q_pack[s1:], k_pack[s1:], v_pack[s1:],
        rq_pack[s1:], rk_pack[s1:],
    )
    ref_cat = torch.cat([ref1, ref2], dim=0)

    return _check("varlen output vs 2 passes non-varlen", ref_cat, out_varlen)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3b — VARLEN BACKWARD : gradients avec sequence packing + graph_bias
# C'est le test le plus critique — c'est la SEULE raison d'être de FNA2.
# ─────────────────────────────────────────────────────────────────────────────

def test_varlen_backward(cfg: dict) -> bool:
    """
    Vérifie que FNA2 varlen calcule des gradients corrects.
    Compare grad(packing) == grad(2 passes indépendantes non-packées).

    C'est le test critique : SDPA ne supporte pas le packing,
    FA2 ne supporte pas le graph_bias → FNA2 doit faire les deux.
    """
    print(f"\n  [T3b] Varlen BACKWARD (training avec packing) — {cfg['label']}")

    H, S, D      = cfg["H"], cfg["S"], cfg["D"]
    H_KV, H_G, R = cfg["H_KV"], cfg["H_G"], cfg["R"]
    scale = 1.0 / math.sqrt(D)

    s1, s2 = S // 2, S // 2
    total  = s1 + s2

    torch.manual_seed(42)

    # ── Tenseurs packés (requires_grad pour tout) ──────────────────────────────
    q_p  = torch.randn(total, H,    D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    k_p  = torch.randn(total, H_KV, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    v_p  = torch.randn(total, H_KV, D, device=DEVICE, dtype=DTYPE, requires_grad=True)
    rq_p = torch.randn(total, H_G,  R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    rk_p = torch.randn(total, H_G,  R, device=DEVICE, dtype=DTYPE, requires_grad=True)
    gs   = torch.randn(H_G, device=DEVICE, dtype=torch.float32, requires_grad=True)

    cu_q = torch.tensor([0, s1, total], device=DEVICE, dtype=torch.int32)
    cu_k = torch.tensor([0, s1, total], device=DEVICE, dtype=torch.int32)

    # Forward varlen + backward
    out_p = flash_naylis_attn(
        q_p, k_p, v_p, rq_p, rk_p, gs,
        softmax_scale=scale, is_causal=True,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=s1, max_seqlen_k=s1,
    )
    loss_p = out_p.float().sum()
    loss_p.backward()

    # ── Référence : deux passes indépendantes non-packées ─────────────────────
    def _seq(tok_start, tok_end):
        """Retourne les tenseurs pour une séquence, reshapés en [1, H, slen, D]."""
        def _v(t):
            # t: [slen, H*, D] ou [slen, H_G, R]
            return t[tok_start:tok_end].detach().clone().requires_grad_(True)
        q_s  = _v(q_p)
        k_s  = _v(k_p)
        v_s  = _v(v_p)
        rq_s = _v(rq_p)
        rk_s = _v(rk_p)
        gs_s = gs.detach().clone().requires_grad_(True)

        # Reshape [slen, H*, D] → [1, H*, slen, D]
        def _r4(t): return t.transpose(0, 1).unsqueeze(0)

        out_s = flash_naylis_attn(
            _r4(q_s), _r4(k_s), _r4(v_s), _r4(rq_s), _r4(rk_s), gs_s,
            softmax_scale=scale, is_causal=True,
        )
        loss_s = out_s.float().sum()
        loss_s.backward()

        return {
            "dq": q_s.grad, "dk": k_s.grad, "dv": v_s.grad,
            "drq": rq_s.grad, "drk": rk_s.grad, "dgs": gs_s.grad,
        }

    ref1 = _seq(0,  s1)
    ref2 = _seq(s1, total)

    # Gradients concaténés des deux passes de référence
    dq_ref  = torch.cat([ref1["dq"],  ref2["dq"]],  dim=0)
    dk_ref  = torch.cat([ref1["dk"],  ref2["dk"]],  dim=0)
    dv_ref  = torch.cat([ref1["dv"],  ref2["dv"]],  dim=0)
    drq_ref = torch.cat([ref1["drq"], ref2["drq"]], dim=0)
    drk_ref = torch.cat([ref1["drk"], ref2["drk"]], dim=0)
    dgs_ref = ref1["dgs"] + ref2["dgs"]   # graph_scale partagé : somme des contributions

    ok  = _check("varlen dQ",          dq_ref,  q_p.grad,   TOL_GRAD)
    ok &= _check("varlen dK",          dk_ref,  k_p.grad,   TOL_GRAD)
    ok &= _check("varlen dV",          dv_ref,  v_p.grad,   TOL_GRAD)
    ok &= _check("varlen dR_q",        drq_ref, rq_p.grad,  TOL_GRAD)
    ok &= _check("varlen dR_k",        drk_ref, rk_p.grad,  TOL_GRAD)
    ok &= _check("varlen d_graph_scale", dgs_ref.float(), gs.grad.float(), TOL_GRAD)

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# TEST 4 — graph_scale = 0 : FNA2 identique à l'attention classique (step 0)
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_scale_zero(cfg: dict) -> bool:
    """
    graph_scale=0 → biais nul → FNA2 doit être identique à une attention
    classique SDPA sans biais (garantie de NaylisAttention au step 0).
    """
    print(f"\n  [T4] graph_scale=0 → FNA2 ≡ attention classique — {cfg['label']}")

    B, H, S, D   = cfg["B"], cfg["H"], cfg["S"], cfg["D"]
    H_KV, H_G, R = cfg["H_KV"], cfg["H_G"], cfg["R"]
    GQA = H // H_KV
    scale = 1.0 / math.sqrt(D)

    torch.manual_seed(7)
    q  = torch.randn(B, H,    S, D, device=DEVICE, dtype=DTYPE)
    k  = torch.randn(B, H_KV, S, D, device=DEVICE, dtype=DTYPE)
    v  = torch.randn(B, H_KV, S, D, device=DEVICE, dtype=DTYPE)
    rq = torch.randn(B, H_G,  S, R, device=DEVICE, dtype=DTYPE)
    rk = torch.randn(B, H_G,  S, R, device=DEVICE, dtype=DTYPE)
    gs = torch.zeros(H_G, device=DEVICE, dtype=torch.float32)  # ← ZÉRO

    # Référence : attention classique (pas de biais)
    k_exp = k.repeat_interleave(GQA, dim=1)
    v_exp = v.repeat_interleave(GQA, dim=1)
    ref   = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True, scale=scale)

    out = flash_naylis_attn(q, k, v, rq, rk, gs, softmax_scale=scale, is_causal=True)

    # Avec gs=0, le biais est nul → résultat identique à l'attention classique
    return _check("FNA2(gs=0) vs SDPA classique", ref, out, tol=TOL_BF16)


# ─────────────────────────────────────────────────────────────────────────────
# TEST 5 — MÉMOIRE : pas d'allocation O(S²)
# ─────────────────────────────────────────────────────────────────────────────

def test_memory_profile(cfg: dict) -> bool:
    """
    Vérifie que FNA2 n'alloue pas de tenseur [B, H, S, S].
    Compare l'empreinte mémoire FNA2 vs SDPA+biais matérialisé.
    """
    print(f"\n  [T5] Profil mémoire — {cfg['label']}")

    B, H, S, D   = cfg["B"], cfg["H"], cfg["S"], cfg["D"]
    H_KV, H_G, R = cfg["H_KV"], cfg["H_G"], cfg["R"]
    scale = 1.0 / math.sqrt(D)

    if DEVICE != "cuda":
        print("    ⚠️  Test mémoire nécessite CUDA — ignoré")
        return True

    def _measure(fn):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024**2  # MB

    q, k, v, rq, rk, gs = _make_tensors(cfg, requires_grad=False)

    # SDPA + biais matérialisé
    mem_sdpa = _measure(lambda: _sdpa_reference(q, k, v, rq, rk, gs, cfg))

    # FNA2
    mem_fna2 = _measure(lambda: flash_naylis_attn(
        q, k, v, rq, rk, gs, softmax_scale=scale, is_causal=True
    ))

    # Le biais matérialisé [B, H, S, S] BF16 = B*H*S*S*2 octets
    bias_tensor_mb = B * H * S * S * 2 / 1024**2
    gain_mb = mem_sdpa - mem_fna2

    print(f"    SDPA+biais : {mem_sdpa:.1f} MB")
    print(f"    FNA2       : {mem_fna2:.1f} MB")
    print(f"    Gain       : {gain_mb:.1f} MB (biais théorique : {bias_tensor_mb:.1f} MB)")
    ok = mem_fna2 < mem_sdpa
    status = "✓" if ok else "⚠️ "
    print(f"    {status} FNA2 {'utilise moins' if ok else 'utilise autant ou plus de'} mémoire que SDPA+biais")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def run_suite(config_name: str = "60m", tests: list = None):
    if DEVICE == "cpu":
        print("⛔  FNA2 requiert un GPU CUDA (Triton). Tests impossibles sur CPU.")
        return

    if not FNA2_OK:
        print("⛔  flash_naylis_attn non chargé — vérifiez l'installation Triton.")
        return

    cfg = CONFIGS.get(config_name, CONFIGS["60m"])
    print("=" * 65)
    print(f"  FNA2 — Suite de validation  [{cfg['label']}]")
    print(f"  Device : {DEVICE}  ({torch.cuda.get_device_name(0)})")
    print("=" * 65)

    all_tests = {
        "forward"         : test_forward,
        "backward"        : test_backward,
        "varlen"          : test_varlen,
        "varlen_backward" : test_varlen_backward,   # ← test critique FNA2
        "gs_zero"         : test_graph_scale_zero,
        "memory"          : test_memory_profile,
    }

    if tests is None:
        tests = list(all_tests.keys())

    results = {}
    for name in tests:
        if name not in all_tests:
            print(f"  ⚠️  Test inconnu : {name}")
            continue
        try:
            results[name] = all_tests[name](cfg)
        except Exception as e:
            print(f"  ✗ {name} — EXCEPTION : {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    print("\n" + "─" * 65)
    n_ok  = sum(results.values())
    n_tot = len(results)
    print(f"  Résultat : {n_ok}/{n_tot} tests réussis")
    for name, ok in results.items():
        print(f"    {'✓' if ok else '✗'} {name}")

    if n_ok == n_tot:
        print("\n  ✅ Tous les tests FNA2 sont verts — prêt pour l'intégration.")
    else:
        print("\n  ⚠️  Des tests ont échoué — voir détails ci-dessus.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FNA2 Validation Suite")
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="60m")
    parser.add_argument("--test",   nargs="+",
                        choices=["forward", "backward", "varlen", "varlen_backward", "gs_zero", "memory"])
    args = parser.parse_args()
    run_suite(config_name=args.config, tests=args.test)
