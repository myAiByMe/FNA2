# FNA2/attention_fna2.py
"""
NaylisAttention avec backend FNA2 intégré.

Ce fichier remplace Core/Attention/attention.py avec :
  - Détection automatique de FNA2 (Triton) au démarrage
  - Backend varlen (sequence packing) AVEC graph_bias via FNA2
  - Fallback propre vers SDPA si FNA2 n'est pas disponible
  - Toute l'API existante préservée (aucun changement de signature)

HIÉRARCHIE DES BACKENDS :
  1. FNA2 varlen  (Triton, fa2_available)  → sequence packing + graph_bias
  2. SDPA         (torch >= 2.0)           → graph_bias via attn_mask [B,H,S,S]
  3. FA std       (flash_attn)             → sans graph_bias (fallback rare)
  4. Manuel       (soft_cap / mask custom) → graph_bias via matmul explicite

GAIN MÉMOIRE FNA2 vs SDPA :
  graph_bias [B, H, S, S] BF16  →  éliminé (O(S²) → O(S))
  R_q / R_k  [B, H_G, S, R]    →  12 MB à S=2048 (vs 150 MB pour le biais)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# DÉTECTION DES BACKENDS
# ─────────────────────────────────────────────────────────────────────────────

_FA_LEVEL       = 0
_FA_VARLEN_FUNC = None
_FA_FUNC        = None
_FNA2_AVAILABLE = False


def _detect_backends():
    global _FA_LEVEL, _FA_VARLEN_FUNC, _FA_FUNC, _FNA2_AVAILABLE

    # ── Détection FNA2 (Triton) ───────────────────────────────────────────────
    # FNA2 est disponible si Triton est installé ET qu'on est sur GPU CUDA
    try:
        import triton  # noqa: F401
        if torch.cuda.is_available():
            # Import du kernel FNA2 (ce fichier est dans le même dossier)
            import importlib, os, sys
            _fna2_dir = os.path.dirname(os.path.abspath(__file__))
            if _fna2_dir not in sys.path:
                sys.path.insert(0, _fna2_dir)
            from flash_naylis_attn import flash_naylis_attn as _fna2_fn, BLOCK_R as _BLOCK_R
            _FNA2_AVAILABLE = True
            _globals = globals()
            _globals['_flash_naylis_attn'] = _fna2_fn
            _globals['_FNA2_BLOCK_R']      = _BLOCK_R
            cap = torch.cuda.get_device_capability()
            arch = "Blackwell SM120" if cap == (12, 0) else f"SM{cap[0]}{cap[1]}"
            print(f"  ⚡ FNA2 (FlashNaylisAttention-2) — Triton — {arch}")
    except (ImportError, Exception) as e:
        _FNA2_AVAILABLE = False
        print(f"  ⚠️  FNA2 non disponible ({e}), fallback SDPA")

    # ── Détection Flash Attention standard (pour le fallback FA std) ──────────
    try:
        import flash_attn
        version = tuple(int(x) for x in flash_attn.__version__.split(".")[:2])
        if version >= (2, 0):
            from flash_attn.flash_attn_interface import (
                flash_attn_func, flash_attn_varlen_func,
            )
            _FA_FUNC        = flash_attn_func
            _FA_VARLEN_FUNC = flash_attn_varlen_func
            _FA_LEVEL       = 2
    except ImportError:
        pass

    # ── Détection SDPA ────────────────────────────────────────────────────────
    if _FA_LEVEL == 0 and hasattr(F, 'scaled_dot_product_attention'):
        _FA_LEVEL = 1
        if not _FNA2_AVAILABLE:
            print("  ⚡ SDPA PyTorch natif (FNA2 absent)")


_detect_backends()

# Références globales injectées par _detect_backends
_flash_naylis_attn = globals().get('_flash_naylis_attn', None)
_FNA2_BLOCK_R      = globals().get('_FNA2_BLOCK_R', 32)


# ─────────────────────────────────────────────────────────────────────────────
# RMSNorm
# ─────────────────────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


# ─────────────────────────────────────────────────────────────────────────────
# RoPE + YaRN
# ─────────────────────────────────────────────────────────────────────────────

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000,
                 use_yarn: bool = False, yarn_scale: float = 1.0,
                 yarn_original_max_len: int = 1024):
        super().__init__()
        self.dim                   = dim
        self.max_seq_len           = max_seq_len
        self.base                  = base
        self.use_yarn              = use_yarn
        self.yarn_scale            = yarn_scale
        self.yarn_original_max_len = yarn_original_max_len

        inv_freq = (self._compute_yarn_frequencies() if use_yarn
                    else 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.register_buffer('inv_freq', inv_freq)
        self._seq_len_cached = None
        self._cos_cached     = None
        self._sin_cached     = None

    def _compute_yarn_frequencies(self):
        freqs         = torch.arange(0, self.dim, 2).float() / self.dim
        inv_freq_base = 1.0 / (self.base ** freqs)
        if self.yarn_scale == 1.0:
            return inv_freq_base
        alpha = self.yarn_scale
        beta  = max(self.dim // 2, int(self.dim * 0.25))
        dims  = torch.arange(0, self.dim, 2).float()
        scale = torch.where(
            dims < beta,
            torch.ones_like(dims),
            1 + (alpha - 1) * (dims - beta) / (self.dim - beta)
        )
        return inv_freq_base / scale

    def _update_cache(self, seq_len: int, device, dtype):
        if (seq_len != self._seq_len_cached
                or self._cos_cached is None
                or self._cos_cached.device != device
                or self._cos_cached.dtype != dtype):
            self._seq_len_cached = seq_len
            t     = torch.arange(seq_len, device=device, dtype=dtype)
            freqs = torch.outer(t, self.inv_freq.to(dtype))
            emb   = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
        return self._cos_cached, self._sin_cached

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor,
                position_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len   = q.shape[2]
        total_len = seq_len + position_offset
        cos, sin  = self._update_cache(total_len, q.device, q.dtype)
        cos = cos[position_offset:position_offset + seq_len][None, None]
        sin = sin[position_offset:position_offset + seq_len][None, None]
        return ((q * cos) + (self._rotate_half(q) * sin),
                (k * cos) + (self._rotate_half(k) * sin))


# ─────────────────────────────────────────────────────────────────────────────
# KV Cache
# ─────────────────────────────────────────────────────────────────────────────

KVCache = Tuple[torch.Tensor, torch.Tensor]


# ─────────────────────────────────────────────────────────────────────────────
# NAYLIS ATTENTION — avec FNA2
# ─────────────────────────────────────────────────────────────────────────────

class NaylisAttention(nn.Module):
    """
    Attention hybride Token-Graph avec backend FNA2.

    Changements vs la version SDPA :
      ① use_fna2 (auto) : si Triton disponible + GPU, utilise FNA2
      ② En mode FNA2 varlen : graph_bias APPLIQUÉ (était désactivé avec FA std)
      ③ Plus de matérialisation du tenseur [B, H, S, S] en mode FNA2
      ④ rel_rank DOIT correspondre à FNA2.BLOCK_R (32 par défaut)

    API identique à l'originale — pas de changement de signature publique.
    """

    def __init__(
        self,
        embed_dim      : int,
        num_heads      : int,
        dropout        : float = 0.0,
        use_rope       : bool  = True,
        max_seq_len    : int   = 2048,
        use_yarn       : bool  = False,
        yarn_scale     : float = 1.0,
        yarn_original_max_len: int = 1024,
        n_kv_heads     : Optional[int] = None,
        use_qk_norm    : bool  = True,
        use_flash_attn : bool  = True,
        soft_cap       : Optional[float] = None,
        rel_rank       : int   = 32,
        sym_heads      : int   = 0,
        vanilla_heads  : int   = 0,
        use_fna2       : Optional[bool] = None,   # None = auto-detect
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        graph_heads = num_heads - vanilla_heads
        assert 0 <= sym_heads <= graph_heads
        assert 0 <= vanilla_heads < num_heads

        # ── Validation FNA2 / rel_rank ────────────────────────────────────────
        # FNA2 exige rel_rank == BLOCK_R (32). Si rel_rank différent, on
        # désactive FNA2 proprement et on revient à SDPA.
        self._fna2_compatible = _FNA2_AVAILABLE and (rel_rank == _FNA2_BLOCK_R)
        if _FNA2_AVAILABLE and rel_rank != _FNA2_BLOCK_R:
            print(f"  ⚠️  FNA2 désactivé : rel_rank={rel_rank} ≠ BLOCK_R={_FNA2_BLOCK_R}. "
                  f"Modifiez BLOCK_R dans flash_naylis_attn.py pour utiliser rel_rank={rel_rank}.")

        if use_fna2 is None:
            self._use_fna2 = self._fna2_compatible and (soft_cap is None)
        else:
            self._use_fna2 = use_fna2 and self._fna2_compatible

        # ── Attributs ────────────────────────────────────────────────────────
        self.embed_dim    = embed_dim
        self.num_heads    = num_heads
        self.head_dim     = embed_dim // num_heads
        self.use_rope     = use_rope
        self.use_qk_norm  = use_qk_norm
        self.soft_cap     = soft_cap
        self.rel_rank     = rel_rank
        self.sym_heads    = sym_heads
        self.vanilla_heads = vanilla_heads
        self.graph_heads  = graph_heads

        self.n_kv_heads         = n_kv_heads if n_kv_heads is not None else num_heads
        assert num_heads % self.n_kv_heads == 0
        self.num_queries_per_kv = num_heads // self.n_kv_heads
        self.kv_dim             = self.n_kv_heads * self.head_dim

        # ── Projections classiques ────────────────────────────────────────────
        self.q_proj   = nn.Linear(embed_dim, embed_dim,   bias=False)
        self.k_proj   = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.v_proj   = nn.Linear(embed_dim, self.kv_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim,   bias=False)
        self.dropout  = nn.Dropout(dropout)

        # ── QK Norm ──────────────────────────────────────────────────────────
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = self.k_norm = None

        # ── RoPE ─────────────────────────────────────────────────────────────
        if use_rope:
            self.rope = RotaryPositionalEmbedding(
                self.head_dim, max_seq_len,
                use_yarn              = use_yarn,
                yarn_scale            = yarn_scale,
                yarn_original_max_len = yarn_original_max_len,
            )
        else:
            self.rope = None

        # ── Naylis : projecteurs relationnels ─────────────────────────────────
        if graph_heads > 0:
            self.rel_q_proj = nn.Linear(embed_dim, graph_heads * rel_rank, bias=False)
            self.rel_k_proj = nn.Linear(embed_dim, graph_heads * rel_rank, bias=False)
            nn.init.normal_(self.rel_q_proj.weight, std=0.01)
            nn.init.normal_(self.rel_k_proj.weight, std=0.01)
            self.graph_scale = nn.Parameter(torch.zeros(graph_heads))
        else:
            self.rel_q_proj  = None
            self.rel_k_proj  = None
            self.graph_scale = None

        # ── Backend SDPA / FA fallback ────────────────────────────────────────
        self._sdpa_ok   = hasattr(F, 'scaled_dot_product_attention')
        self._fa_level  = _FA_LEVEL if use_flash_attn else 0
        self._fa_varlen = _FA_VARLEN_FUNC
        self._fa_func   = _FA_FUNC

    def _attn_scale(self) -> float:
        if (self.use_rope and self.rope is not None
                and self.rope.use_yarn and self.rope.yarn_scale > 1.0):
            return math.sqrt(self.rope.yarn_scale) / math.sqrt(self.head_dim)
        return 1.0 / math.sqrt(self.head_dim)

    def _compute_rq_rk(
        self,
        x    : torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule R_q et R_k depuis x.

        Mode normal  : retourne [B, H_G, S, R]
        Toujours en BF16 — contrainte non-négociable NaylisAttention.
        """
        B, S, _ = x.shape
        H_g, R  = self.graph_heads, self.rel_rank

        R_q = self.rel_q_proj(x).view(B, S, H_g, R).permute(0, 2, 1, 3).to(dtype)
        R_k = self.rel_k_proj(x).view(B, S, H_g, R).permute(0, 2, 1, 3).to(dtype)

        return R_q.contiguous(), R_k.contiguous()

    def _compute_rq_rk_varlen(
        self,
        x    : torch.Tensor,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calcule R_q et R_k en mode varlen (x est [total_tokens, embed_dim]).
        Retourne [total_tokens, H_G, R].
        """
        total_tokens, _ = x.shape
        H_g, R = self.graph_heads, self.rel_rank

        R_q = self.rel_q_proj(x).view(total_tokens, H_g, R).to(dtype)
        R_k = self.rel_k_proj(x).view(total_tokens, H_g, R).to(dtype)

        return R_q.contiguous(), R_k.contiguous()

    def _compute_graph_bias_sdpa(
        self,
        x    : torch.Tensor,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Fallback SDPA : matérialise le tenseur [B, H, S, S] complet.
        Utilisé uniquement quand FNA2 n'est pas disponible.

        Contrainte : dtype doit être torch.bfloat16.
        """
        assert dtype == torch.bfloat16, \
            "graph_bias doit rester en BF16 — ne jamais passer float8 ou float4"

        B, S, _ = x.shape
        H_g, R  = self.graph_heads, self.rel_rank

        R_q = self.rel_q_proj(x).view(B, S, H_g, R).permute(0, 2, 1, 3)
        R_k = self.rel_k_proj(x).view(B, S, H_g, R).permute(0, 2, 1, 3)

        # [B, H_g, S, S]
        graph_bias = torch.matmul(R_q, R_k.transpose(-2, -1))

        # Symétrisation (sym_heads éliminés après Run 5 — gardé pour compatibilité)
        if self.sym_heads > 0:
            sym = graph_bias[:, :self.sym_heads]
            sym = (sym + sym.transpose(-2, -1)) * 0.5
            graph_bias = torch.cat([sym, graph_bias[:, self.sym_heads:]], dim=1)

        scale      = self.graph_scale.view(1, H_g, 1, 1)
        graph_bias = (scale * graph_bias).to(dtype)

        # Pad vanilla_heads avec zéros
        if self.vanilla_heads > 0:
            pad        = torch.zeros(B, self.vanilla_heads, S, S, dtype=dtype, device=x.device)
            graph_bias = torch.cat([graph_bias, pad], dim=1)

        return graph_bias.contiguous()

    # ─────────────────────────────────────────────────────────────────────────
    # FORWARD
    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x            : torch.Tensor,
        mask         : Optional[torch.Tensor] = None,
        past_kv      : Optional[KVCache]      = None,
        use_kv_cache : bool                   = False,
        cu_seqlens_q : Optional[torch.Tensor] = None,
        cu_seqlens_k : Optional[torch.Tensor] = None,
        max_seqlen_q : Optional[int]          = None,
        max_seqlen_k : Optional[int]          = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:

        # Mode varlen : x est [total_tokens, embed_dim], pas de batch dim
        is_varlen = (cu_seqlens_q is not None)

        if is_varlen:
            return self._forward_varlen(
                x, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k
            )

        B, S, _ = x.shape
        scale   = self._attn_scale()

        # ── Projections QKV ──────────────────────────────────────────────────
        q = self.q_proj(x).view(B, S, self.num_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        position_offset = past_kv[0].shape[2] if past_kv is not None else 0
        if self.use_rope and self.rope is not None:
            q, k = self.rope(q, k, position_offset=position_offset)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv_cache: Optional[KVCache] = (k, v) if use_kv_cache else None

        # ── Cast BF16 ────────────────────────────────────────────────────────
        orig_dtype = q.dtype
        if q.dtype == torch.float32:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        is_causal = (S > 1 and past_kv is None)

        # En decode (S=1 + KV cache) : pas de graph_bias (R_k porte sur S_total)
        _use_graph = (self.graph_heads > 0) and (S > 1 or past_kv is None)

        # ── Backend FNA2 ─────────────────────────────────────────────────────
        if self._use_fna2 and _use_graph and self.soft_cap is None and past_kv is None:
            # FNA2 gère GQA nativement — pas de repeat_interleave
            r_q, r_k = self._compute_rq_rk(x, torch.bfloat16)

            # graph_scale doit être en float32 (scalaire learnable AdamW)
            gs = self.graph_scale.to(torch.float32)

            output = _flash_naylis_attn(
                q, k, v, r_q, r_k, gs,
                softmax_scale = scale,
                is_causal     = is_causal,
            )

        # ── Backend SDPA (fallback ou decode) ────────────────────────────────
        elif self._sdpa_ok and self.soft_cap is None:
            # GQA : expand K/V
            if self.n_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

            if _use_graph:
                graph_bias = self._compute_graph_bias_sdpa(x, q.dtype)
            else:
                graph_bias = None

            if is_causal and graph_bias is not None:
                S_q, S_kv = q.shape[2], k.shape[2]
                causal_mask = torch.triu(
                    torch.full((S_q, S_kv), float('-inf'),
                               device=q.device, dtype=q.dtype),
                    diagonal=1 + (past_kv[0].shape[2] if past_kv else 0),
                )
                attn_mask = (graph_bias + causal_mask.unsqueeze(0).unsqueeze(0)).contiguous()
                output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, is_causal=False,
                    dropout_p=self.dropout.p if self.training else 0.0, scale=scale,
                )
            elif graph_bias is not None:
                output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=graph_bias.contiguous(), is_causal=False,
                    dropout_p=self.dropout.p if self.training else 0.0, scale=scale,
                )
            else:
                output = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, is_causal=is_causal,
                    dropout_p=self.dropout.p if self.training else 0.0, scale=scale,
                )

        # ── Backend FA std fallback ───────────────────────────────────────────
        elif self._fa_func is not None and self.soft_cap is None and mask is None:
            if self.n_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            output = self._fa_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=scale, causal=is_causal,
            )
            output = output.transpose(1, 2)

        # ── Backend manuel ────────────────────────────────────────────────────
        else:
            if self.n_kv_heads != self.num_heads:
                k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
                v = v.repeat_interleave(self.num_queries_per_kv, dim=1)
            graph_bias = self._compute_graph_bias_sdpa(x, q.dtype) if _use_graph else None

            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if self.soft_cap is not None:
                scores = self.soft_cap * torch.tanh(scores / self.soft_cap)
            if graph_bias is not None:
                scores = scores + graph_bias
            if S > 1 and past_kv is None:
                if mask is not None:
                    scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                else:
                    causal_bool = torch.triu(
                        torch.ones(S, k.shape[2], device=q.device, dtype=torch.bool), diagonal=1
                    )
                    scores = scores.masked_fill(causal_bool.unsqueeze(0).unsqueeze(0), float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            if self.training and self.dropout.p > 0:
                attn_weights = self.dropout(attn_weights)
            output = torch.matmul(attn_weights, v)

        # ── Reshape + projection ─────────────────────────────────────────────
        output = output.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        if output.dtype != orig_dtype:
            output = output.to(orig_dtype)
        if output.dtype != self.out_proj.weight.dtype:
            output = output.to(self.out_proj.weight.dtype)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, new_kv_cache

    # ─────────────────────────────────────────────────────────────────────────
    # FORWARD VARLEN — sequence packing avec FNA2
    # ─────────────────────────────────────────────────────────────────────────

    def _forward_varlen(
        self,
        x            : torch.Tensor,          # [total_tokens, embed_dim]
        cu_seqlens_q : torch.Tensor,
        cu_seqlens_k : torch.Tensor,
        max_seqlen_q : Optional[int],
        max_seqlen_k : Optional[int],
    ) -> Tuple[torch.Tensor, None]:
        """
        Forward en mode varlen (sequence packing).

        AVANT FNA2 : graph_bias était désactivé ici (varlen ne supporte pas
                     attn_mask dense). Le backend FA std était utilisé sans biais.

        APRÈS FNA2 : le biais est calculé et appliqué tuile par tuile.
                     Aucun tenseur [total_tokens, H, S, S] n'est matérialisé.
        """
        total_tokens, embed_dim = x.shape
        scale = self._attn_scale()

        # Projections QKV → [total_tokens, H, D]
        q = self.q_proj(x).view(total_tokens, self.num_heads,  self.head_dim)
        k = self.k_proj(x).view(total_tokens, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(total_tokens, self.n_kv_heads, self.head_dim)

        if self.use_qk_norm:
            # QK norm sur les shapes varlen
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE varlen — à implémenter avec positions continues si besoin
        # Pour l'instant : RoPE appliqué séquence par séquence via cu_seqlens
        if self.use_rope and self.rope is not None:
            q, k = self._apply_rope_varlen(q, k, cu_seqlens_q)

        orig_dtype = q.dtype
        if q.dtype == torch.float32:
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            v = v.to(torch.bfloat16)

        _msl_q = max_seqlen_q if max_seqlen_q is not None else total_tokens
        _msl_k = max_seqlen_k if max_seqlen_k is not None else total_tokens

        # ── FNA2 varlen : graph_bias appliqué pour la première fois ──────────
        if self._use_fna2 and self.graph_heads > 0:
            r_q, r_k = self._compute_rq_rk_varlen(x, torch.bfloat16)
            gs = self.graph_scale.to(torch.float32)

            output = _flash_naylis_attn(
                q, k, v, r_q, r_k, gs,
                softmax_scale = scale,
                is_causal     = True,
                cu_seqlens_q  = cu_seqlens_q,
                cu_seqlens_k  = cu_seqlens_k,
                max_seqlen_q  = _msl_q,
                max_seqlen_k  = _msl_k,
            )

        # ── Fallback FA std varlen (sans graph_bias, comme avant FNA2) ────────
        elif self._fa_varlen is not None:
            output = self._fa_varlen(
                q, k, v,
                cu_seqlens_q, cu_seqlens_k,
                _msl_q, _msl_k,
                dropout_p     = self.dropout.p if self.training else 0.0,
                softmax_scale = scale,
                causal        = True,
            )

        else:
            raise RuntimeError(
                "Mode varlen nécessite FNA2 (Triton) ou flash_attn >= 2.0. "
                "Installez Triton ou flash-attn, ou désactivez use_varlen."
            )

        # ── Projection sortie ────────────────────────────────────────────────
        output = output.reshape(total_tokens, self.embed_dim)
        if output.dtype != orig_dtype:
            output = output.to(orig_dtype)
        if output.dtype != self.out_proj.weight.dtype:
            output = output.to(self.out_proj.weight.dtype)
        output = self.out_proj(output)
        output = self.dropout(output)

        return output, None

    def _apply_rope_varlen(
        self,
        q: torch.Tensor,   # [total_tokens, H, D]
        k: torch.Tensor,   # [total_tokens, H_KV, D]
        cu_seqlens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applique RoPE en mode varlen : les positions repartent à 0 pour
        chaque nouvelle séquence packée.
        """
        q_out = torch.empty_like(q)
        k_out = torch.empty_like(k)

        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        offset  = 0
        for slen in seqlens:
            slen = int(slen)
            q_s = q[offset:offset + slen].transpose(0, 1).unsqueeze(0)  # [1, H, slen, D]
            k_s = k[offset:offset + slen].transpose(0, 1).unsqueeze(0)  # [1, H_KV, slen, D]

            # RoPE positions 0..slen-1 (début de séquence)
            q_r, k_r = self.rope(q_s, k_s, position_offset=0)

            q_out[offset:offset + slen] = q_r.squeeze(0).transpose(0, 1)
            k_out[offset:offset + slen] = k_r.squeeze(0).transpose(0, 1)
            offset += slen

        return q_out, k_out
