# FNA2/flash_naylis_attn.py
"""
FlashNaylisAttention-2 (FNA2)
==============================
Kernel Triton dérivé de FlashAttention-2 qui fusionne rel_q_proj / rel_k_proj
directement dans la boucle de tiling — calcul du biais tuile par tuile.

GAIN MÉMOIRE :
  Avant FNA2 : graph_bias [B, H, S, S] en BF16  →  O(S²) VRAM  (~150 MB à S=2048)
  Après FNA2 : R_q [B, H_G, S, R] + R_k [B, H_G, S, R]  →  O(S) VRAM  (~12 MB)

ARCHITECTURE :
  ┌ Pour chaque tuile Q_i (BLOCK_M lignes de Q) :
  │  ┌ Pour chaque tuile K_j / V_j (BLOCK_N colonnes) :
  │  │   S_ij  = Q_i @ K_j^T  * scale            ← dot-product classique
  │  │   rk_j  = R_k[j*BN:(j+1)*BN]              ← chargé DRAM, O(BN*R)
  │  │   B_ij  = gs * R_q_i @ rk_j^T             ← biais tuile [BM, BN]
  │  │   S_ij += B_ij                             ← fusion
  │  │   Appliquer masque causal + softmax en ligne (Flash)
  │  │   acc  += softmax(S_ij) @ V_j
  │  └─
  │  Stocker Out_i, LSE_i
  └─

COMPATIBILITÉ : SM90 (H100), SM120 (RTX Pro 6000 Blackwell)
REQUIS : triton >= 2.2, torch >= 2.0

TÊTES :
  - Têtes graph  [0 .. H_G-1] : biais asymétrique B[i,j] = gs * R_q_i . R_k_j^T
  - Têtes vanilla [H_G .. H-1] : attention classique — R_q/R_k non chargés
  - Têtes symétriques : NON supportées (éliminées après Run 5, section 5.1 du paper)

GQA : kv_head = q_head // gqa_ratio (natif dans le kernel, aucun repeat_interleave)

VARLEN : cu_seqlens supporté via un kernel dédié (_fna2_fwd_varlen_kernel).
"""
import math
from typing import Optional

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE BLOCS
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_M   = 128   # lignes de Q par tuile
BLOCK_N   = 64    # colonnes de K/V par tuile
BLOCK_R   = 32    # rel_rank (doit correspondre à celui de NaylisAttention)


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL FORWARD — STANDARD (non-varlen)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fna2_fwd_kernel(
    # ── Tenseurs d'entrée ───────────────────────────────────────────────────
    Q,            # [B, H,    S, D]
    K,            # [B, H_KV, S, D]
    V,            # [B, H_KV, S, D]
    R_q,          # [B, H_G,  S, R]  — uniquement pour les têtes graph
    R_k,          # [B, H_G,  S, R]  — uniquement pour les têtes graph
    graph_scale,  # [H_G]             — par tête graph, init=0 → transformer classique
    # ── Sortie ──────────────────────────────────────────────────────────────
    Out,          # [B, H, S, D]
    Lse,          # [B, H, S]        — log-sum-exp stocké pour le backward
    # ── Strides Q [B, H, S, D] ───────────────────────────────────────────
    stride_qb, stride_qh, stride_qs, stride_qd,
    # ── Strides K [B, H_KV, S, D] ────────────────────────────────────────
    stride_kb, stride_kh, stride_ks, stride_kd,
    # ── Strides V [B, H_KV, S, D] ────────────────────────────────────────
    stride_vb, stride_vh, stride_vs, stride_vd,
    # ── Strides Out [B, H, S, D] ─────────────────────────────────────────
    stride_ob, stride_oh, stride_os, stride_od,
    # ── Strides R_q / R_k [B, H_G, S, R] ────────────────────────────────
    stride_rqb, stride_rqh, stride_rqs, stride_rqr,
    stride_rkb, stride_rkh, stride_rks, stride_rkr,
    # ── Strides Lse [B, H, S] ────────────────────────────────────────────
    stride_lseb, stride_lseh,
    # ── Dimensions ───────────────────────────────────────────────────────
    B, H, H_G, H_KV, S, GQA_RATIO,
    softmax_scale,
    # ── Constexpr ────────────────────────────────────────────────────────
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr,
    BLOCK_D : tl.constexpr,
    BLOCK_R : tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    # ── IDs de programme ─────────────────────────────────────────────────────
    pid_m  = tl.program_id(0)    # indice de la tuile Q
    pid_bh = tl.program_id(1)    # indice (batch × head)

    off_b  = pid_bh  // H
    off_h  = pid_bh  %  H
    off_kv = off_h   // GQA_RATIO   # tête KV pour GQA

    q_start = pid_m * BLOCK_M

    offs_m = q_start + tl.arange(0, BLOCK_M)   # positions dans S pour cette tuile Q
    offs_d = tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)

    # ── Pointeurs de base ─────────────────────────────────────────────────────
    Q_base  = Q   + off_b * stride_qb  + off_h  * stride_qh
    K_base  = K   + off_b * stride_kb  + off_kv * stride_kh
    V_base  = V   + off_b * stride_vb  + off_kv * stride_vh
    O_base  = Out + off_b * stride_ob  + off_h  * stride_oh
    Lse_base = Lse + off_b * stride_lseb + off_h * stride_lseh
    Rq_base = R_q + off_b * stride_rqb + off_h  * stride_rqh
    Rk_base = R_k + off_b * stride_rkb + off_h  * stride_rkh

    # ── Chargement de la tuile Q [BLOCK_M, BLOCK_D] ───────────────────────────
    q_mask = offs_m[:, None] < S
    q_ptrs = Q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ── Chargement de la tuile R_q [BLOCK_M, BLOCK_R] (têtes graph seulement) ─
    is_graph_head = off_h < H_G
    if is_graph_head:
        rq_ptrs = Rq_base + offs_m[:, None] * stride_rqs + offs_r[None, :] * stride_rqr
        rq = tl.load(rq_ptrs, mask=offs_m[:, None] < S, other=0.0).to(tl.float32)
        gs = tl.load(graph_scale + off_h)
    else:
        # Triton nécessite des variables définies dans toutes les branches
        rq = tl.zeros([BLOCK_M, BLOCK_R], dtype=tl.float32)
        gs = tl.zeros([1], dtype=tl.float32)

    # ── Accumulateurs softmax en ligne (algo FlashAttention-2) ────────────────
    m_i  = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # ── Bornes de la boucle K/V (masque causal) ───────────────────────────────
    if IS_CAUSAL:
        # Pour la tuile Q commençant à q_start, les tuiles K au-delà de
        # q_start + BLOCK_M - 1 sont entièrement masquées → on peut les ignorer.
        kv_hi = tl.cdiv(tl.minimum(S, q_start + BLOCK_M), BLOCK_N)
    else:
        kv_hi = tl.cdiv(S, BLOCK_N)

    # ── Boucle sur les tuiles K/V ─────────────────────────────────────────────
    for tile_n in range(0, kv_hi):
        start_n = tile_n * BLOCK_N
        offs_n  = start_n + tl.arange(0, BLOCK_N)

        kn_valid = offs_n[None, :] < S

        # K chargé transposé [BLOCK_D, BLOCK_N] pour Q @ K^T
        k_ptrs = K_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks
        k = tl.load(k_ptrs, mask=kn_valid, other=0.0)   # [BLOCK_D, BLOCK_N]

        # Scores Q @ K^T [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * softmax_scale

        # ── Biais Naylis (têtes graph seulement) ─────────────────────────────
        if is_graph_head:
            rk_ptrs = Rk_base + offs_n[:, None] * stride_rks + offs_r[None, :] * stride_rkr
            rk = tl.load(rk_ptrs, mask=offs_n[:, None] < S, other=0.0).to(tl.float32)
            # B_ij = R_q_i @ R_k_j^T   [BLOCK_M, BLOCK_N]
            bias = tl.dot(rq, tl.trans(rk))
            qk   = qk + gs * bias

        # ── Masque causal ─────────────────────────────────────────────────────
        if IS_CAUSAL:
            causal_ok = offs_m[:, None] >= offs_n[None, :]
            qk = tl.where(causal_ok, qk, float('-inf'))

        # ── Masque OOB ───────────────────────────────────────────────────────
        in_range = (offs_m[:, None] < S) & (offs_n[None, :] < S)
        qk = tl.where(in_range, qk, float('-inf'))

        # ── Softmax en ligne : mise à jour de m_i, l_i, acc ──────────────────
        m_ij  = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)                        # [BLOCK_M]
        p     = tl.exp(qk.to(tl.float32) - m_new[:, None]) # [BLOCK_M, BLOCK_N]

        # V chargé [BLOCK_N, BLOCK_D]
        v_ptrs = V_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < S, other=0.0)

        # Accumulation : acc = alpha * acc + p @ v
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # ── Normalisation finale ───────────────────────────────────────────────────
    acc = acc / (l_i[:, None] + 1e-6)

    # ── Stockage de LSE [BLOCK_M] ─────────────────────────────────────────────
    lse = m_i + tl.log(l_i + 1e-6)
    lse_ptrs = Lse_base + offs_m
    tl.store(lse_ptrs, lse, mask=offs_m < S)

    # ── Stockage de la sortie [BLOCK_M, BLOCK_D] ──────────────────────────────
    o_ptrs = O_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < S)


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL FORWARD — VARLEN (sequence packing avec cu_seqlens)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fna2_fwd_varlen_kernel(
    # ── Tenseurs d'entrée (séquences packées, dim B=1) ──────────────────────
    Q,           # [total_tokens, H, D]    — layout varlen (pas de batch dim)
    K,           # [total_tokens, H_KV, D]
    V,           # [total_tokens, H_KV, D]
    R_q,         # [total_tokens, H_G, R]
    R_k,         # [total_tokens, H_G, R]
    graph_scale, # [H_G]
    # ── Offsets de séquences ────────────────────────────────────────────────
    cu_seqlens_q,  # [N_seq + 1]  — offsets cumulatifs (query side)
    cu_seqlens_k,  # [N_seq + 1]  — offsets cumulatifs (key side)
    max_seqlen_q,  # scalaire
    max_seqlen_k,  # scalaire
    # ── Sortie ──────────────────────────────────────────────────────────────
    Out,         # [total_tokens, H, D]
    Lse,         # [total_tokens, H]
    # ── Strides Q/K/V/Out : [total_tokens, H*, D] → strides sur dim 0 et 2 ─
    stride_qs, stride_qh, stride_qd,
    stride_ks, stride_kh, stride_kd,
    stride_vs, stride_vh, stride_vd,
    stride_os, stride_oh, stride_od,
    stride_rqs, stride_rqh, stride_rqr,
    stride_rks, stride_rkh, stride_rkr,
    # ── Dimensions ──────────────────────────────────────────────────────────
    N_seq, H, H_G, H_KV, GQA_RATIO,
    softmax_scale,
    # ── Constexpr ───────────────────────────────────────────────────────────
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr,
    BLOCK_D : tl.constexpr,
    BLOCK_R : tl.constexpr,
):
    # pid_m  = indice de la tuile Q au sein de la séquence
    # pid_h  = indice de tête
    # pid_sq = indice de la séquence dans le batch
    pid_m  = tl.program_id(0)
    pid_h  = tl.program_id(1)
    pid_sq = tl.program_id(2)

    # Offsets de séquence pour Q et K
    seq_start_q = tl.load(cu_seqlens_q + pid_sq)
    seq_end_q   = tl.load(cu_seqlens_q + pid_sq + 1)
    seq_start_k = tl.load(cu_seqlens_k + pid_sq)
    seq_end_k   = tl.load(cu_seqlens_k + pid_sq + 1)

    seq_len_q   = seq_end_q - seq_start_q
    seq_len_k   = seq_end_k - seq_start_k

    q_start = pid_m * BLOCK_M
    if q_start >= seq_len_q:
        return   # tuile hors de la séquence

    off_h  = pid_h
    off_kv = off_h // GQA_RATIO

    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)

    # Positions absolues dans le flux packé
    abs_m = seq_start_q + offs_m   # positions absolues pour Q

    # ── Bases de pointeurs ────────────────────────────────────────────────────
    Q_base  = Q   + off_h  * stride_qh
    K_base  = K   + off_kv * stride_kh
    V_base  = V   + off_kv * stride_vh
    O_base  = Out + off_h  * stride_oh
    Rq_base = R_q + off_h  * stride_rqh
    Rk_base = R_k + off_h  * stride_rkh

    # ── Chargement de Q [BLOCK_M, BLOCK_D] ───────────────────────────────────
    q_mask = offs_m[:, None] < seq_len_q
    q_ptrs = Q_base + abs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # ── R_q (têtes graph) ─────────────────────────────────────────────────────
    is_gh = off_h < H_G
    if is_gh:
        rq_ptrs = Rq_base + abs_m[:, None] * stride_rqs + offs_r[None, :] * stride_rqr
        rq = tl.load(rq_ptrs, mask=offs_m[:, None] < seq_len_q, other=0.0).to(tl.float32)
        gs = tl.load(graph_scale + off_h)
    else:
        rq = tl.zeros([BLOCK_M, BLOCK_R], dtype=tl.float32)
        gs = tl.zeros([1], dtype=tl.float32)

    # ── Accumulateurs ─────────────────────────────────────────────────────────
    m_i  = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i  = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Boucle sur les tuiles K/V (attention causale intra-séquence)
    kv_hi = tl.cdiv(tl.minimum(seq_len_k, q_start + BLOCK_M), BLOCK_N)

    for tile_n in range(0, kv_hi):
        start_n    = tile_n * BLOCK_N
        offs_n     = start_n + tl.arange(0, BLOCK_N)
        abs_n      = seq_start_k + offs_n
        kn_valid   = offs_n[None, :] < seq_len_k

        # K [BLOCK_D, BLOCK_N] (transposé)
        k_ptrs = K_base + offs_d[:, None] * stride_kd + abs_n[None, :] * stride_ks
        k = tl.load(k_ptrs, mask=kn_valid, other=0.0)

        qk = tl.dot(q, k) * softmax_scale

        # Biais Naylis
        if is_gh:
            rk_ptrs = Rk_base + abs_n[:, None] * stride_rks + offs_r[None, :] * stride_rkr
            rk = tl.load(rk_ptrs, mask=offs_n[:, None] < seq_len_k, other=0.0).to(tl.float32)
            bias = tl.dot(rq, tl.trans(rk))
            qk   = qk + gs * bias

        # Masque causal intra-séquence
        causal_ok = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_ok, qk, float('-inf'))

        in_range = (offs_m[:, None] < seq_len_q) & (offs_n[None, :] < seq_len_k)
        qk = tl.where(in_range, qk, float('-inf'))

        m_ij  = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(qk.to(tl.float32) - m_new[:, None])

        v_ptrs = V_base + abs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < seq_len_k, other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v).to(tl.float32)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    acc = acc / (l_i[:, None] + 1e-6)
    lse = m_i + tl.log(l_i + 1e-6)

    # Stockage LSE [BLOCK_M]
    lse_ptrs = Lse + abs_m * H + off_h
    tl.store(lse_ptrs, lse, mask=offs_m < seq_len_q)

    # Stockage sortie [BLOCK_M, BLOCK_D]
    o_ptrs = O_base + abs_m[:, None] * stride_os + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < seq_len_q)


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL BACKWARD — Prétraitement : delta = sum(dO * O, dim=-1)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fna2_bwd_preprocess_kernel(
    O, dO,     # [B, H, S, D]
    Delta,     # [B, H, S]     — sortie
    stride_ob, stride_oh, stride_os, stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    stride_db, stride_dh,
    B, H, S,
    BLOCK_D: tl.constexpr,
):
    pid_s  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_b = pid_bh // H
    off_h = pid_bh %  H

    offs_d = tl.arange(0, BLOCK_D)
    mask_d = offs_d < BLOCK_D

    O_base  = O  + off_b * stride_ob  + off_h * stride_oh  + pid_s * stride_os
    dO_base = dO + off_b * stride_dob + off_h * stride_doh + pid_s * stride_dos

    o  = tl.load(O_base  + offs_d * stride_od,  mask=mask_d & (pid_s < S), other=0.0).to(tl.float32)
    do = tl.load(dO_base + offs_d * stride_dod, mask=mask_d & (pid_s < S), other=0.0).to(tl.float32)

    delta = tl.sum(o * do, axis=0)
    tl.store(Delta + off_b * stride_db + off_h * stride_dh + pid_s, delta, mask=pid_s < S)


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL BACKWARD — dK, dV, dR_k
# Itère sur les tuiles Q pour chaque tuile K fixée
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fna2_bwd_dkdv_kernel(
    # Entrées forward
    Q, K, V, R_q, R_k, graph_scale,
    Out, Lse, Delta,
    dO,
    # Sorties backward
    dK, dV, dR_k,
    # dgraph_scale : accumulé atomiquement par head
    d_gs,
    # Strides [B, H, S, D]
    stride_qb,  stride_qh,  stride_qs,  stride_qd,
    stride_kb,  stride_kh,  stride_ks,  stride_kd,
    stride_vb,  stride_vh,  stride_vs,  stride_vd,
    stride_ob,  stride_oh,  stride_os,  stride_od,
    stride_dob, stride_doh, stride_dos, stride_dod,
    # Strides [B, H_G, S, R]
    stride_rqb, stride_rqh, stride_rqs, stride_rqr,
    stride_rkb, stride_rkh, stride_rks, stride_rkr,
    # Strides delta / Lse [B, H, S]
    stride_db, stride_dh,
    stride_lb, stride_lh,
    # Strides sorties [B, H*, S, D] et [B, H_G, S, R]
    stride_dkb, stride_dkh, stride_dks, stride_dkd,
    stride_dvb, stride_dvh, stride_dvs, stride_dvd,
    stride_drkb, stride_drkh, stride_drks, stride_drkr,
    # Dims
    B, H, H_G, H_KV, S, GQA_RATIO,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_n  = tl.program_id(0)   # indice de la tuile K/V
    pid_bh = tl.program_id(1)   # (batch, head Q)

    off_b  = pid_bh // H
    off_h  = pid_bh %  H
    off_kv = off_h  // GQA_RATIO

    kv_start = pid_n * BLOCK_N
    offs_n   = kv_start + tl.arange(0, BLOCK_N)
    offs_d   = tl.arange(0, BLOCK_D)
    offs_r   = tl.arange(0, BLOCK_R)

    is_gh = off_h < H_G

    # ── Chargement de la tuile K [BLOCK_D, BLOCK_N] (transposé) ──────────────
    K_base = K + off_b * stride_kb + off_kv * stride_kh
    k_ptrs = K_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks
    k      = tl.load(k_ptrs, mask=offs_n[None, :] < S, other=0.0)   # [BLOCK_D, BLOCK_N]

    # ── Chargement de la tuile V [BLOCK_N, BLOCK_D] ───────────────────────────
    V_base = V + off_b * stride_vb + off_kv * stride_vh
    v_ptrs = V_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
    v      = tl.load(v_ptrs, mask=offs_n[:, None] < S, other=0.0)   # [BLOCK_N, BLOCK_D]

    # ── R_k pour cette tuile [BLOCK_N, BLOCK_R] ───────────────────────────────
    if is_gh:
        Rk_base = R_k + off_b * stride_rkb + off_h * stride_rkh
        rk_ptrs = Rk_base + offs_n[:, None] * stride_rks + offs_r[None, :] * stride_rkr
        rk = tl.load(rk_ptrs, mask=offs_n[:, None] < S, other=0.0).to(tl.float32)
        gs = tl.load(graph_scale + off_h)
    else:
        rk = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)
        gs = tl.zeros([1], dtype=tl.float32)

    # ── Accumulateurs ─────────────────────────────────────────────────────────
    dK_acc   = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dV_acc   = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dRk_acc  = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)
    dgs_acc  = tl.zeros([1], dtype=tl.float32)

    # Borne de la boucle Q (causal : Q[i] peut voir K[j] seulement si i >= j)
    if IS_CAUSAL:
        q_lo = kv_start   # premières tuiles Q qui peuvent avoir des interactions
    else:
        q_lo = 0
    n_q_tiles = tl.cdiv(S, BLOCK_M)

    Q_base   = Q   + off_b * stride_qb  + off_h  * stride_qh
    O_base   = Out + off_b * stride_ob  + off_h  * stride_oh
    dO_base  = dO  + off_b * stride_dob + off_h  * stride_doh
    Rq_base  = R_q + off_b * stride_rqb + off_h  * stride_rqh
    Lse_base = Lse + off_b * stride_lb  + off_h  * stride_lh
    Del_base = Delta + off_b * stride_db + off_h * stride_dh

    for tile_m in range(0, n_q_tiles):
        start_m = tile_m * BLOCK_M
        offs_m  = start_m + tl.arange(0, BLOCK_M)

        if IS_CAUSAL and start_m + BLOCK_M <= kv_start:
            continue   # tuile Q entièrement avant la tuile K → masquée

        qm_valid = offs_m[:, None] < S

        # Q, dO, Out, LSE, Delta pour cette tuile Q
        q  = tl.load(Q_base  + offs_m[:, None] * stride_qs  + offs_d[None, :] * stride_qd,
                     mask=qm_valid, other=0.0)
        do = tl.load(dO_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                     mask=qm_valid, other=0.0)
        o  = tl.load(O_base  + offs_m[:, None] * stride_os  + offs_d[None, :] * stride_od,
                     mask=qm_valid, other=0.0)
        lse   = tl.load(Lse_base + offs_m, mask=offs_m < S, other=0.0)
        delta = tl.load(Del_base + offs_m, mask=offs_m < S, other=0.0)

        if is_gh:
            rq = tl.load(Rq_base + offs_m[:, None] * stride_rqs + offs_r[None, :] * stride_rqr,
                         mask=offs_m[:, None] < S, other=0.0).to(tl.float32)
        else:
            rq = tl.zeros([BLOCK_M, BLOCK_R], dtype=tl.float32)

        # ── Recalcul de P_ij depuis LSE (Flash-Attention style) ───────────────
        # scores = Q @ K^T * scale  [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k) * softmax_scale

        if is_gh:
            bias = tl.dot(rq, tl.trans(rk))
            qk   = qk + gs * bias

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))

        qk = tl.where((offs_m[:, None] < S) & (offs_n[None, :] < S), qk, float('-inf'))

        # P_ij = exp(qk - LSE_i)
        p = tl.exp(qk.to(tl.float32) - lse[:, None])   # [BLOCK_M, BLOCK_N]

        # ── Gradient softmax : dS = P * (dP - delta) ─────────────────────────
        # dP_ij = dO_i @ V_j^T   [BLOCK_M, BLOCK_N]
        dp = tl.dot(do.to(v.dtype), tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None])                  # [BLOCK_M, BLOCK_N]

        # ── Gradients K, V ───────────────────────────────────────────────────
        # dV_j += P_ij^T @ dO_i
        dV_acc  += tl.dot(tl.trans(p).to(do.dtype), do).to(tl.float32)
        # dK_j += dS_ij^T @ Q_i * scale   (dK en layout [BLOCK_N, BLOCK_D])
        dK_acc  += tl.dot(tl.trans(ds).to(q.dtype), q).to(tl.float32) * softmax_scale

        # ── Gradients R_k (têtes graph) ───────────────────────────────────────
        # d(B_ij) = ds  →  d(gs * R_q_i @ R_k_j^T) = ds
        # dR_k_j += dS_ij^T @ R_q_i * gs   [BLOCK_N, BLOCK_R]
        if is_gh:
            dRk_acc  += tl.dot(tl.trans(ds).to(rq.dtype), rq).to(tl.float32) * gs
            # dgraph_scale[h] += sum(dS * bias)
            raw_bias  = tl.dot(rq, tl.trans(rk))        # [BLOCK_M, BLOCK_N]
            dgs_acc  += tl.sum(ds * raw_bias)

    # ── Écriture des sorties ───────────────────────────────────────────────────
    dK_base = dK + off_b * stride_dkb + off_kv * stride_dkh
    dV_base = dV + off_b * stride_dvb + off_kv * stride_dvh

    dK_ptrs = dK_base + offs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd
    dV_ptrs = dV_base + offs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd
    tl.atomic_add(dK_ptrs, dK_acc.to(dK.dtype.element_ty), mask=offs_n[:, None] < S)
    tl.atomic_add(dV_ptrs, dV_acc.to(dV.dtype.element_ty), mask=offs_n[:, None] < S)

    if is_gh:
        dRk_base = dR_k + off_b * stride_drkb + off_h * stride_drkh
        dRk_ptrs = dRk_base + offs_n[:, None] * stride_drks + offs_r[None, :] * stride_drkr
        tl.atomic_add(dRk_ptrs, dRk_acc.to(dR_k.dtype.element_ty), mask=offs_n[:, None] < S)
        tl.atomic_add(d_gs + off_h, dgs_acc.to(tl.float32))


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL BACKWARD — dQ, dR_q
# Itère sur les tuiles K pour chaque tuile Q fixée
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fna2_bwd_dq_kernel(
    # Entrées
    Q, K, V, R_q, R_k, graph_scale,
    Lse, Delta, dO,
    # Sorties
    dQ, dR_q,
    # Strides [B, H, S, D]
    stride_qb,  stride_qh,  stride_qs,  stride_qd,
    stride_kb,  stride_kh,  stride_ks,  stride_kd,
    stride_vb,  stride_vh,  stride_vs,  stride_vd,
    stride_dob, stride_doh, stride_dos, stride_dod,
    # Strides [B, H_G, S, R]
    stride_rqb, stride_rqh, stride_rqs, stride_rqr,
    stride_rkb, stride_rkh, stride_rks, stride_rkr,
    # Strides delta / Lse
    stride_db, stride_dh,
    stride_lb, stride_lh,
    # Strides sorties dQ [B, H, S, D], dR_q [B, H_G, S, R]
    stride_dqb, stride_dqh, stride_dqs, stride_dqd,
    stride_drqb, stride_drqh, stride_drqs, stride_drqr,
    # Dims
    B, H, H_G, H_KV, S, GQA_RATIO,
    softmax_scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_R: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    pid_m  = tl.program_id(0)   # indice de la tuile Q
    pid_bh = tl.program_id(1)   # (batch, head)

    off_b  = pid_bh // H
    off_h  = pid_bh %  H
    off_kv = off_h  // GQA_RATIO

    q_start = pid_m * BLOCK_M
    offs_m  = q_start + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)
    offs_r  = tl.arange(0, BLOCK_R)

    is_gh = off_h < H_G

    qm_valid = offs_m[:, None] < S

    Q_base   = Q   + off_b * stride_qb  + off_h  * stride_qh
    K_base   = K   + off_b * stride_kb  + off_kv * stride_kh
    V_base   = V   + off_b * stride_vb  + off_kv * stride_vh
    dO_base  = dO  + off_b * stride_dob + off_h  * stride_doh
    Rq_base  = R_q + off_b * stride_rqb + off_h  * stride_rqh
    Rk_base  = R_k + off_b * stride_rkb + off_h  * stride_rkh
    Lse_base = Lse + off_b * stride_lb  + off_h  * stride_lh
    Del_base = Delta + off_b * stride_db + off_h * stride_dh

    # Chargement Q, dO, LSE, Delta pour la tuile Q fixée
    q  = tl.load(Q_base  + offs_m[:, None] * stride_qs  + offs_d[None, :] * stride_qd,
                 mask=qm_valid, other=0.0)
    do = tl.load(dO_base + offs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                 mask=qm_valid, other=0.0)
    lse   = tl.load(Lse_base + offs_m, mask=offs_m < S, other=0.0)
    delta = tl.load(Del_base + offs_m, mask=offs_m < S, other=0.0)

    if is_gh:
        rq = tl.load(Rq_base + offs_m[:, None] * stride_rqs + offs_r[None, :] * stride_rqr,
                     mask=offs_m[:, None] < S, other=0.0).to(tl.float32)
        gs = tl.load(graph_scale + off_h)
    else:
        rq = tl.zeros([BLOCK_M, BLOCK_R], dtype=tl.float32)
        gs = tl.zeros([1], dtype=tl.float32)

    dQ_acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    dRq_acc = tl.zeros([BLOCK_M, BLOCK_R], dtype=tl.float32)

    # Causal : on ne regarde que les tuiles K accessibles
    if IS_CAUSAL:
        kv_hi = tl.cdiv(tl.minimum(S, q_start + BLOCK_M), BLOCK_N)
    else:
        kv_hi = tl.cdiv(S, BLOCK_N)

    for tile_n in range(0, kv_hi):
        start_n = tile_n * BLOCK_N
        offs_n  = start_n + tl.arange(0, BLOCK_N)

        kn_valid = offs_n[None, :] < S

        # K [BLOCK_D, BLOCK_N]
        k_ptrs = K_base + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_ks
        k = tl.load(k_ptrs, mask=kn_valid, other=0.0)

        # V [BLOCK_N, BLOCK_D]
        v_ptrs = V_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_n[:, None] < S, other=0.0)

        if is_gh:
            rk = tl.load(Rk_base + offs_n[:, None] * stride_rks + offs_r[None, :] * stride_rkr,
                         mask=offs_n[:, None] < S, other=0.0).to(tl.float32)
        else:
            rk = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)

        # Recalcul P_ij
        qk = tl.dot(q, k) * softmax_scale

        if is_gh:
            bias = tl.dot(rq, tl.trans(rk))
            qk   = qk + gs * bias

        if IS_CAUSAL:
            qk = tl.where(offs_m[:, None] >= offs_n[None, :], qk, float('-inf'))

        qk = tl.where((offs_m[:, None] < S) & kn_valid, qk, float('-inf'))

        p = tl.exp(qk.to(tl.float32) - lse[:, None])

        dp = tl.dot(do.to(v.dtype), tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta[:, None])

        # dQ_i += dS_ij @ K_j^T * scale   (K en [BLOCK_D, BLOCK_N] → transposé [BLOCK_N, BLOCK_D])
        dQ_acc += tl.dot(ds.to(k.dtype), tl.trans(k)).to(tl.float32) * softmax_scale

        # dR_q_i += dS_ij @ R_k_j * gs   [BLOCK_M, BLOCK_R]
        if is_gh:
            dRq_acc += tl.dot(ds.to(rk.dtype), rk).to(tl.float32) * gs

    # ── Écriture ──────────────────────────────────────────────────────────────
    dQ_base = dQ + off_b * stride_dqb + off_h * stride_dqh
    dQ_ptrs = dQ_base + offs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd
    tl.atomic_add(dQ_ptrs, dQ_acc.to(dQ.dtype.element_ty), mask=qm_valid)

    if is_gh:
        dRq_base = dR_q + off_b * stride_drqb + off_h * stride_drqh
        dRq_ptrs = dRq_base + offs_m[:, None] * stride_drqs + offs_r[None, :] * stride_drqr
        tl.atomic_add(dRq_ptrs, dRq_acc.to(dR_q.dtype.element_ty), mask=offs_m[:, None] < S)


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS PYTHON D'APPEL DES KERNELS
# ─────────────────────────────────────────────────────────────────────────────

def _fna2_forward(
    q: torch.Tensor,           # [B, H, S, D]
    k: torch.Tensor,           # [B, H_KV, S, D]
    v: torch.Tensor,           # [B, H_KV, S, D]
    r_q: torch.Tensor,         # [B, H_G, S, R]
    r_k: torch.Tensor,         # [B, H_G, S, R]
    graph_scale: torch.Tensor, # [H_G]
    softmax_scale: float,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lance le kernel forward FNA2, retourne (output, lse)."""

    B, H, S, D = q.shape
    H_KV       = k.shape[1]
    H_G        = r_q.shape[1]
    R          = r_q.shape[3]
    GQA_RATIO  = H // H_KV

    assert H_G == r_k.shape[1], "r_q et r_k doivent avoir le même nombre de têtes graph"
    assert R == BLOCK_R, f"rel_rank doit être {BLOCK_R}, reçu {R}"
    assert D in (64, 128), f"head_dim doit être 64 ou 128 (reçu {D})"

    out = torch.empty_like(q)
    lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(S, BLOCK_M), B * H)

    _fna2_fwd_kernel[grid](
        q, k, v, r_q, r_k, graph_scale,
        out, lse,
        # Strides Q
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # Strides K
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Strides V
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        # Strides Out
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        # Strides R_q
        r_q.stride(0), r_q.stride(1), r_q.stride(2), r_q.stride(3),
        # Strides R_k
        r_k.stride(0), r_k.stride(1), r_k.stride(2), r_k.stride(3),
        # Strides Lse
        lse.stride(0), lse.stride(1),
        # Dims
        B, H, H_G, H_KV, S, GQA_RATIO,
        softmax_scale,
        # Constexpr
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D, BLOCK_R=BLOCK_R,
        IS_CAUSAL=is_causal,
        num_warps=4 if D <= 64 else 8,
        num_stages=2,
    )
    return out, lse


def _fna2_forward_varlen(
    q: torch.Tensor,           # [total_tokens, H, D]
    k: torch.Tensor,           # [total_tokens, H_KV, D]
    v: torch.Tensor,           # [total_tokens, H_KV, D]
    r_q: torch.Tensor,         # [total_tokens, H_G, R]
    r_k: torch.Tensor,         # [total_tokens, H_G, R]
    graph_scale: torch.Tensor, # [H_G]
    cu_seqlens_q: torch.Tensor,  # [N_seq + 1]
    cu_seqlens_k: torch.Tensor,  # [N_seq + 1]
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lance le kernel forward FNA2 en mode varlen (sequence packing)."""

    total_q, H, D = q.shape
    H_KV          = k.shape[1]
    H_G           = r_q.shape[1]
    R             = r_q.shape[2]
    GQA_RATIO     = H // H_KV
    N_seq         = cu_seqlens_q.shape[0] - 1

    assert R == BLOCK_R, f"rel_rank doit être {BLOCK_R}, reçu {R}"

    out = torch.empty_like(q)
    # LSE stocké comme [total_tokens, H] pour la compatibilité varlen
    lse = torch.empty(total_q, H, device=q.device, dtype=torch.float32)

    grid = (triton.cdiv(max_seqlen_q, BLOCK_M), H, N_seq)

    _fna2_fwd_varlen_kernel[grid](
        q, k, v, r_q, r_k, graph_scale,
        cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        out, lse,
        # Strides [total_tokens, H, D] → stride(0)=H*D, stride(1)=D, stride(2)=1
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        # Strides R [total_tokens, H_G, R]
        r_q.stride(0), r_q.stride(1), r_q.stride(2),
        r_k.stride(0), r_k.stride(1), r_k.stride(2),
        N_seq, H, H_G, H_KV, GQA_RATIO,
        softmax_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D, BLOCK_R=BLOCK_R,
        num_warps=4 if D <= 64 else 8,
        num_stages=2,
    )
    return out, lse


def _fna2_backward(
    dout: torch.Tensor,
    q, k, v, r_q, r_k, graph_scale,
    out, lse,
    softmax_scale: float,
    is_causal: bool,
) -> tuple[torch.Tensor, ...]:
    """Lance les kernels backward FNA2. Retourne (dq, dk, dv, dr_q, dr_k, d_gs)."""

    B, H, S, D = q.shape
    H_KV       = k.shape[1]
    H_G        = r_q.shape[1]
    R          = r_q.shape[3]
    GQA_RATIO  = H // H_KV

    # ── Prétraitement : delta = (dO * O).sum(-1) ─────────────────────────────
    delta = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
    pre_grid = (S, B * H)
    _fna2_bwd_preprocess_kernel[pre_grid](
        out, dout, delta,
        out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        delta.stride(0), delta.stride(1),
        B, H, S,
        BLOCK_D=D,
    )

    # ── Allocation des gradients ──────────────────────────────────────────────
    dq    = torch.zeros_like(q)
    dk    = torch.zeros_like(k)
    dv    = torch.zeros_like(v)
    dr_q  = torch.zeros_like(r_q)
    dr_k  = torch.zeros_like(r_k)
    d_gs  = torch.zeros(H_G, device=q.device, dtype=torch.float32)

    common_args = dict(
        softmax_scale=softmax_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D, BLOCK_R=BLOCK_R,
        IS_CAUSAL=is_causal,
        num_warps=4 if D <= 64 else 8,
        num_stages=2,
    )

    # ── Kernel dK, dV, dR_k ───────────────────────────────────────────────────
    dkdv_grid = (triton.cdiv(S, BLOCK_N), B * H)
    _fna2_bwd_dkdv_kernel[dkdv_grid](
        q, k, v, r_q, r_k, graph_scale,
        out, lse, delta, dout,
        dk, dv, dr_k, d_gs,
        q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
        k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
        v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
        out.stride(0),  out.stride(1),  out.stride(2),  out.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        r_q.stride(0),  r_q.stride(1),  r_q.stride(2),  r_q.stride(3),
        r_k.stride(0),  r_k.stride(1),  r_k.stride(2),  r_k.stride(3),
        delta.stride(0), delta.stride(1),
        lse.stride(0),   lse.stride(1),
        dk.stride(0),   dk.stride(1),   dk.stride(2),   dk.stride(3),
        dv.stride(0),   dv.stride(1),   dv.stride(2),   dv.stride(3),
        dr_k.stride(0), dr_k.stride(1), dr_k.stride(2), dr_k.stride(3),
        B, H, H_G, H_KV, S, GQA_RATIO,
        **common_args,
    )

    # ── Kernel dQ, dR_q ───────────────────────────────────────────────────────
    dq_grid = (triton.cdiv(S, BLOCK_M), B * H)
    _fna2_bwd_dq_kernel[dq_grid](
        q, k, v, r_q, r_k, graph_scale,
        lse, delta, dout,
        dq, dr_q,
        q.stride(0),    q.stride(1),    q.stride(2),    q.stride(3),
        k.stride(0),    k.stride(1),    k.stride(2),    k.stride(3),
        v.stride(0),    v.stride(1),    v.stride(2),    v.stride(3),
        dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
        r_q.stride(0),  r_q.stride(1),  r_q.stride(2),  r_q.stride(3),
        r_k.stride(0),  r_k.stride(1),  r_k.stride(2),  r_k.stride(3),
        delta.stride(0), delta.stride(1),
        lse.stride(0),   lse.stride(1),
        dq.stride(0),   dq.stride(1),   dq.stride(2),   dq.stride(3),
        dr_q.stride(0), dr_q.stride(1), dr_q.stride(2), dr_q.stride(3),
        B, H, H_G, H_KV, S, GQA_RATIO,
        **common_args,
    )

    return dq, dk, dv, dr_q, dr_k, d_gs.to(graph_scale.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL BACKWARD VARLEN — Prétraitement delta (layout varlen)
# delta[abs, h] = sum(dO[abs, h, :] * O[abs, h, :])
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fna2_bwd_preprocess_varlen_kernel(
    O, dO,     # [total_tokens, H, D]
    Delta,     # [total_tokens, H]  — flat: Delta[abs * H + off_h]
    stride_os, stride_oh, stride_od,
    stride_dos, stride_doh, stride_dod,
    total_tokens, H,
    BLOCK_D : tl.constexpr,
):
    pid_s = tl.program_id(0)   # token absolu
    pid_h = tl.program_id(1)   # tête

    if pid_s >= total_tokens:
        return

    offs_d = tl.arange(0, BLOCK_D)

    o  = tl.load(O  + pid_s * stride_os + pid_h * stride_oh + offs_d * stride_od).to(tl.float32)
    do = tl.load(dO + pid_s * stride_dos + pid_h * stride_doh + offs_d * stride_dod).to(tl.float32)

    delta = tl.sum(o * do, axis=0)
    tl.store(Delta + pid_s * H + pid_h, delta)


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL BACKWARD VARLEN — dK, dV, dR_k
# Chaque programme fixe une tuile K et itère sur les tuiles Q de la même séquence.
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fna2_bwd_dkdv_varlen_kernel(
    # Entrées forward
    Q, K, V, R_q, R_k, graph_scale,
    Out, dO,
    Lse, Delta,           # [total_tokens, H]  flat: [abs * H + off_h]
    cu_seqlens_q,         # [N_seq + 1]
    cu_seqlens_k,         # [N_seq + 1]
    # Sorties (atomic_add — plusieurs programmes Q écrivent sur les mêmes K)
    dK, dV, dR_k, d_gs,
    # Strides [total_tokens, H*, D]
    stride_qs,  stride_qh,  stride_qd,
    stride_ks,  stride_kh,  stride_kd,
    stride_vs,  stride_vh,  stride_vd,
    stride_dos, stride_doh, stride_dod,
    stride_rqs, stride_rqh, stride_rqr,
    stride_rks, stride_rkh, stride_rkr,
    stride_dks, stride_dkh, stride_dkd,
    stride_dvs, stride_dvh, stride_dvd,
    stride_drks, stride_drkh, stride_drkr,
    # Dimensions
    H, H_G, H_KV, GQA_RATIO,
    softmax_scale,
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr,
    BLOCK_D : tl.constexpr,
    BLOCK_R : tl.constexpr,
):
    pid_n  = tl.program_id(0)   # tuile K dans la séquence
    pid_h  = tl.program_id(1)   # tête
    pid_sq = tl.program_id(2)   # séquence

    seq_start_q = tl.load(cu_seqlens_q + pid_sq)
    seq_end_q   = tl.load(cu_seqlens_q + pid_sq + 1)
    seq_start_k = tl.load(cu_seqlens_k + pid_sq)
    seq_end_k   = tl.load(cu_seqlens_k + pid_sq + 1)
    seq_len_q   = seq_end_q - seq_start_q
    seq_len_k   = seq_end_k - seq_start_k

    kv_start = pid_n * BLOCK_N
    if kv_start >= seq_len_k:
        return

    off_h   = pid_h
    off_kv  = off_h // GQA_RATIO
    is_gh   = off_h < H_G

    offs_n = kv_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)
    abs_n  = seq_start_k + offs_n        # positions absolues dans le flux packé
    kn_ok  = offs_n < seq_len_k

    # ── Chargement de la tuile K [BLOCK_D, BLOCK_N] (transposée) ─────────────
    K_base = K + off_kv * stride_kh
    k = tl.load(K_base + offs_d[:, None] * stride_kd + abs_n[None, :] * stride_ks,
                mask=kn_ok[None, :], other=0.0)

    # ── Chargement de la tuile V [BLOCK_N, BLOCK_D] ───────────────────────────
    V_base = V + off_kv * stride_vh
    v = tl.load(V_base + abs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                mask=kn_ok[:, None], other=0.0)

    # ── Chargement de R_k [BLOCK_N, BLOCK_R] ─────────────────────────────────
    if is_gh:
        Rk_base = R_k + off_h * stride_rkh
        rk = tl.load(Rk_base + abs_n[:, None] * stride_rks + offs_r[None, :] * stride_rkr,
                     mask=kn_ok[:, None], other=0.0).to(tl.float32)
        gs = tl.load(graph_scale + off_h)
    else:
        rk = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)
        gs = tl.zeros([1], dtype=tl.float32)

    # ── Accumulateurs ─────────────────────────────────────────────────────────
    dK_acc  = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dV_acc  = tl.zeros([BLOCK_N, BLOCK_D], dtype=tl.float32)
    dRk_acc = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)
    dgs_acc = tl.zeros([1],               dtype=tl.float32)

    Q_base  = Q  + off_h * stride_qh
    dO_base = dO + off_h * stride_doh
    Rq_base = R_q + off_h * stride_rqh

    n_q_tiles = tl.cdiv(seq_len_q, BLOCK_M)

    for tile_m in range(0, n_q_tiles):
        start_m = tile_m * BLOCK_M
        offs_m  = start_m + tl.arange(0, BLOCK_M)
        abs_m   = seq_start_q + offs_m
        qm_ok   = offs_m < seq_len_q

        # Masque causal : la tuile Q doit avoir au moins un token >= kv_start
        if start_m + BLOCK_M <= kv_start:
            continue

        # Charger Q, dO
        q  = tl.load(Q_base  + abs_m[:, None] * stride_qs  + offs_d[None, :] * stride_qd,
                     mask=qm_ok[:, None], other=0.0)
        do = tl.load(dO_base + abs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                     mask=qm_ok[:, None], other=0.0)

        lse_v   = tl.load(Lse   + abs_m * H + off_h, mask=qm_ok, other=0.0)
        delta_v = tl.load(Delta + abs_m * H + off_h, mask=qm_ok, other=0.0)

        if is_gh:
            rq = tl.load(Rq_base + abs_m[:, None] * stride_rqs + offs_r[None, :] * stride_rqr,
                         mask=qm_ok[:, None], other=0.0).to(tl.float32)
        else:
            rq = tl.zeros([BLOCK_M, BLOCK_R], dtype=tl.float32)

        # Recalcul des scores (même formule que le forward)
        qk = tl.dot(q, k) * softmax_scale
        if is_gh:
            bias = tl.dot(rq, tl.trans(rk))
            qk   = qk + gs * bias

        causal_ok = offs_m[:, None] >= offs_n[None, :]
        in_range  = qm_ok[:, None] & kn_ok[None, :]
        qk = tl.where(causal_ok & in_range, qk, float('-inf'))

        p  = tl.exp(qk.to(tl.float32) - lse_v[:, None])   # [BLOCK_M, BLOCK_N]

        # dP = dO @ V^T
        dp = tl.dot(do.to(v.dtype), tl.trans(v)).to(tl.float32)   # [BLOCK_M, BLOCK_N]
        ds = p * (dp - delta_v[:, None])                            # [BLOCK_M, BLOCK_N]

        # Accumulation
        dV_acc  += tl.dot(tl.trans(p).to(do.dtype), do).to(tl.float32)
        dK_acc  += tl.dot(tl.trans(ds).to(q.dtype), q).to(tl.float32) * softmax_scale

        if is_gh:
            dRk_acc += tl.dot(tl.trans(ds).to(rq.dtype), rq).to(tl.float32) * gs
            raw_bias = tl.dot(rq, tl.trans(rk))
            dgs_acc += tl.sum(ds * raw_bias)

    # ── Écriture atomique (plusieurs Q-tiles peuvent contribuer) ─────────────
    dK_base = dK + off_kv * stride_dkh
    dV_base = dV + off_kv * stride_dvh
    tl.atomic_add(dK_base + abs_n[:, None] * stride_dks + offs_d[None, :] * stride_dkd,
                  dK_acc.to(dK.dtype.element_ty), mask=kn_ok[:, None])
    tl.atomic_add(dV_base + abs_n[:, None] * stride_dvs + offs_d[None, :] * stride_dvd,
                  dV_acc.to(dV.dtype.element_ty), mask=kn_ok[:, None])

    if is_gh:
        dRk_base = dR_k + off_h * stride_drkh
        tl.atomic_add(dRk_base + abs_n[:, None] * stride_drks + offs_r[None, :] * stride_drkr,
                      dRk_acc.to(dR_k.dtype.element_ty), mask=kn_ok[:, None])
        tl.atomic_add(d_gs + off_h, dgs_acc.to(tl.float32))


# ─────────────────────────────────────────────────────────────────────────────
# KERNEL BACKWARD VARLEN — dQ, dR_q
# Chaque programme fixe une tuile Q et itère sur les tuiles K visibles (causal).
# dQ est écrit en direct (chaque position Q est gérée par un seul programme).
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _fna2_bwd_dq_varlen_kernel(
    Q, K, V, R_q, R_k, graph_scale,
    Lse, Delta, dO,
    cu_seqlens_q,
    cu_seqlens_k,
    dQ, dR_q,
    # Strides [total_tokens, H*, D]
    stride_qs,  stride_qh,  stride_qd,
    stride_ks,  stride_kh,  stride_kd,
    stride_vs,  stride_vh,  stride_vd,
    stride_dos, stride_doh, stride_dod,
    stride_rqs, stride_rqh, stride_rqr,
    stride_rks, stride_rkh, stride_rkr,
    stride_dqs,  stride_dqh,  stride_dqd,
    stride_drqs, stride_drqh, stride_drqr,
    # Dimensions
    H, H_G, H_KV, GQA_RATIO,
    softmax_scale,
    BLOCK_M : tl.constexpr,
    BLOCK_N : tl.constexpr,
    BLOCK_D : tl.constexpr,
    BLOCK_R : tl.constexpr,
):
    pid_m  = tl.program_id(0)   # tuile Q dans la séquence
    pid_h  = tl.program_id(1)   # tête
    pid_sq = tl.program_id(2)   # séquence

    seq_start_q = tl.load(cu_seqlens_q + pid_sq)
    seq_end_q   = tl.load(cu_seqlens_q + pid_sq + 1)
    seq_start_k = tl.load(cu_seqlens_k + pid_sq)
    seq_end_k   = tl.load(cu_seqlens_k + pid_sq + 1)
    seq_len_q   = seq_end_q - seq_start_q
    seq_len_k   = seq_end_k - seq_start_k

    q_start = pid_m * BLOCK_M
    if q_start >= seq_len_q:
        return

    off_h  = pid_h
    off_kv = off_h // GQA_RATIO
    is_gh  = off_h < H_G

    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_r = tl.arange(0, BLOCK_R)
    abs_m  = seq_start_q + offs_m
    qm_ok  = offs_m < seq_len_q

    # ── Chargement des données Q fixées ───────────────────────────────────────
    Q_base  = Q  + off_h  * stride_qh
    dO_base = dO + off_h  * stride_doh
    Rq_base = R_q + off_h * stride_rqh

    q  = tl.load(Q_base  + abs_m[:, None] * stride_qs  + offs_d[None, :] * stride_qd,
                 mask=qm_ok[:, None], other=0.0)
    do = tl.load(dO_base + abs_m[:, None] * stride_dos + offs_d[None, :] * stride_dod,
                 mask=qm_ok[:, None], other=0.0)
    lse_v   = tl.load(Lse   + abs_m * H + off_h, mask=qm_ok, other=0.0)
    delta_v = tl.load(Delta + abs_m * H + off_h, mask=qm_ok, other=0.0)

    if is_gh:
        rq = tl.load(Rq_base + abs_m[:, None] * stride_rqs + offs_r[None, :] * stride_rqr,
                     mask=qm_ok[:, None], other=0.0).to(tl.float32)
        gs = tl.load(graph_scale + off_h)
    else:
        rq = tl.zeros([BLOCK_M, BLOCK_R], dtype=tl.float32)
        gs = tl.zeros([1], dtype=tl.float32)

    dQ_acc  = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)
    dRq_acc = tl.zeros([BLOCK_M, BLOCK_R], dtype=tl.float32)

    K_base  = K + off_kv * stride_kh
    V_base  = V + off_kv * stride_vh
    Rk_base = R_k + off_h * stride_rkh

    # Causal : la tuile Q voit les tuiles K jusqu'à q_start + BLOCK_M
    kv_hi = tl.cdiv(tl.minimum(seq_len_k, q_start + BLOCK_M), BLOCK_N)

    for tile_n in range(0, kv_hi):
        start_n = tile_n * BLOCK_N
        offs_n  = start_n + tl.arange(0, BLOCK_N)
        abs_n   = seq_start_k + offs_n
        kn_ok   = offs_n < seq_len_k

        k = tl.load(K_base + offs_d[:, None] * stride_kd + abs_n[None, :] * stride_ks,
                    mask=kn_ok[None, :], other=0.0)
        v = tl.load(V_base + abs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
                    mask=kn_ok[:, None], other=0.0)

        if is_gh:
            rk = tl.load(Rk_base + abs_n[:, None] * stride_rks + offs_r[None, :] * stride_rkr,
                         mask=kn_ok[:, None], other=0.0).to(tl.float32)
        else:
            rk = tl.zeros([BLOCK_N, BLOCK_R], dtype=tl.float32)

        qk = tl.dot(q, k) * softmax_scale
        if is_gh:
            bias = tl.dot(rq, tl.trans(rk))
            qk   = qk + gs * bias

        causal_ok = offs_m[:, None] >= offs_n[None, :]
        in_range  = qm_ok[:, None] & kn_ok[None, :]
        qk = tl.where(causal_ok & in_range, qk, float('-inf'))

        p  = tl.exp(qk.to(tl.float32) - lse_v[:, None])

        dp = tl.dot(do.to(v.dtype), tl.trans(v)).to(tl.float32)
        ds = p * (dp - delta_v[:, None])

        dQ_acc  += tl.dot(ds.to(k.dtype), tl.trans(k)).to(tl.float32) * softmax_scale

        if is_gh:
            dRq_acc += tl.dot(ds.to(rk.dtype), rk).to(tl.float32) * gs

    # ── Écriture directe (chaque programme gère des positions Q uniques) ──────
    dQ_base = dQ + off_h * stride_dqh
    tl.store(dQ_base + abs_m[:, None] * stride_dqs + offs_d[None, :] * stride_dqd,
             dQ_acc.to(dQ.dtype.element_ty), mask=qm_ok[:, None])

    if is_gh:
        dRq_base = dR_q + off_h * stride_drqh
        tl.store(dRq_base + abs_m[:, None] * stride_drqs + offs_r[None, :] * stride_drqr,
                 dRq_acc.to(dR_q.dtype.element_ty), mask=qm_ok[:, None])


# ─────────────────────────────────────────────────────────────────────────────
# LANCEUR PYTHON — backward varlen
# ─────────────────────────────────────────────────────────────────────────────

def _fna2_backward_varlen(
    dout: torch.Tensor,
    q, k, v, r_q, r_k, graph_scale,
    out, lse,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
) -> tuple[torch.Tensor, ...]:
    """Lance les kernels backward FNA2 varlen. Retourne (dq, dk, dv, dr_q, dr_k, d_gs)."""

    total_q, H, D = q.shape
    H_KV          = k.shape[1]
    H_G           = r_q.shape[1] if r_q is not None else 0
    N_seq         = cu_seqlens_q.shape[0] - 1
    GQA_RATIO     = H // H_KV

    dout = dout.contiguous()

    # ── Prétraitement : delta[abs * H + off_h] = sum(dO * O) ─────────────────
    delta = torch.empty(total_q, H, device=q.device, dtype=torch.float32)
    pre_grid = (total_q, H)
    _fna2_bwd_preprocess_varlen_kernel[pre_grid](
        out, dout, delta,
        out.stride(0),  out.stride(1),  out.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        total_q, H,
        BLOCK_D=D,
    )

    # ── Allocation des gradients ───────────────────────────────────────────────
    dq   = torch.zeros_like(q)
    dk   = torch.zeros_like(k)
    dv   = torch.zeros_like(v)
    dr_q = torch.zeros_like(r_q)
    dr_k = torch.zeros_like(r_k)
    d_gs = torch.zeros(H_G, device=q.device, dtype=torch.float32)

    common = dict(
        H=H, H_G=H_G, H_KV=H_KV, GQA_RATIO=GQA_RATIO,
        softmax_scale=softmax_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=D, BLOCK_R=BLOCK_R,
        num_warps=4 if D <= 64 else 8,
        num_stages=2,
    )

    # ── Kernel dK, dV, dR_k : grid=(cdiv(max_seqlen_k, BLOCK_N), H, N_seq) ──
    dkdv_grid = (triton.cdiv(max_seqlen_k, BLOCK_N), H, N_seq)
    _fna2_bwd_dkdv_varlen_kernel[dkdv_grid](
        q, k, v, r_q, r_k, graph_scale,
        out, dout,
        lse, delta,
        cu_seqlens_q, cu_seqlens_k,
        dk, dv, dr_k, d_gs,
        q.stride(0),   q.stride(1),   q.stride(2),
        k.stride(0),   k.stride(1),   k.stride(2),
        v.stride(0),   v.stride(1),   v.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        r_q.stride(0), r_q.stride(1), r_q.stride(2),
        r_k.stride(0), r_k.stride(1), r_k.stride(2),
        dk.stride(0),  dk.stride(1),  dk.stride(2),
        dv.stride(0),  dv.stride(1),  dv.stride(2),
        dr_k.stride(0), dr_k.stride(1), dr_k.stride(2),
        **common,
    )

    # ── Kernel dQ, dR_q : grid=(cdiv(max_seqlen_q, BLOCK_M), H, N_seq) ──────
    dq_grid = (triton.cdiv(max_seqlen_q, BLOCK_M), H, N_seq)
    _fna2_bwd_dq_varlen_kernel[dq_grid](
        q, k, v, r_q, r_k, graph_scale,
        lse, delta, dout,
        cu_seqlens_q, cu_seqlens_k,
        dq, dr_q,
        q.stride(0),    q.stride(1),    q.stride(2),
        k.stride(0),    k.stride(1),    k.stride(2),
        v.stride(0),    v.stride(1),    v.stride(2),
        dout.stride(0), dout.stride(1), dout.stride(2),
        r_q.stride(0),  r_q.stride(1),  r_q.stride(2),
        r_k.stride(0),  r_k.stride(1),  r_k.stride(2),
        dq.stride(0),   dq.stride(1),   dq.stride(2),
        dr_q.stride(0), dr_q.stride(1), dr_q.stride(2),
        **common,
    )

    return dq, dk, dv, dr_q, dr_k, d_gs.to(graph_scale.dtype)


# ─────────────────────────────────────────────────────────────────────────────
# AUTOGRAD FUNCTION — interface PyTorch autograd
# ─────────────────────────────────────────────────────────────────────────────

class FlashNaylisAttnFunc(torch.autograd.Function):
    """
    Wrapper torch.autograd.Function pour FNA2.
    Interface propre pour l'intégration dans NaylisAttention.
    """

    @staticmethod
    def forward(
        ctx,
        q, k, v, r_q, r_k, graph_scale,
        softmax_scale: float,
        is_causal: bool,
        cu_seqlens_q=None,
        cu_seqlens_k=None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
    ):
        use_varlen = (cu_seqlens_q is not None)

        if use_varlen:
            out, lse = _fna2_forward_varlen(
                q, k, v, r_q, r_k, graph_scale,
                cu_seqlens_q, cu_seqlens_k,
                max_seqlen_q, max_seqlen_k,
                softmax_scale,
            )
        else:
            out, lse = _fna2_forward(
                q, k, v, r_q, r_k, graph_scale,
                softmax_scale, is_causal,
            )

        ctx.save_for_backward(q, k, v, r_q, r_k, graph_scale, out, lse)
        ctx.softmax_scale = softmax_scale
        ctx.is_causal     = is_causal
        ctx.use_varlen    = use_varlen
        # Infos varlen conservées pour le backward
        ctx.cu_seqlens_q  = cu_seqlens_q
        ctx.cu_seqlens_k  = cu_seqlens_k
        ctx.max_seqlen_q  = max_seqlen_q
        ctx.max_seqlen_k  = max_seqlen_k

        return out

    @staticmethod
    def backward(ctx, dout):
        q, k, v, r_q, r_k, graph_scale, out, lse = ctx.saved_tensors

        dout = dout.contiguous()

        if ctx.use_varlen:
            dq, dk, dv, dr_q, dr_k, d_gs = _fna2_backward_varlen(
                dout, q, k, v, r_q, r_k, graph_scale,
                out, lse,
                ctx.cu_seqlens_q, ctx.cu_seqlens_k,
                ctx.max_seqlen_q, ctx.max_seqlen_k,
                ctx.softmax_scale,
            )
        else:
            dq, dk, dv, dr_q, dr_k, d_gs = _fna2_backward(
                dout, q, k, v, r_q, r_k, graph_scale,
                out, lse,
                ctx.softmax_scale, ctx.is_causal,
            )

        # 12 retours : les args non-tenseur retournent None
        return dq, dk, dv, dr_q, dr_k, d_gs, None, None, None, None, None, None


# ─────────────────────────────────────────────────────────────────────────────
# API PUBLIQUE
# ─────────────────────────────────────────────────────────────────────────────

def flash_naylis_attn(
    q            : torch.Tensor,
    k            : torch.Tensor,
    v            : torch.Tensor,
    r_q          : torch.Tensor,
    r_k          : torch.Tensor,
    graph_scale  : torch.Tensor,
    softmax_scale: Optional[float] = None,
    is_causal    : bool = True,
    cu_seqlens_q : Optional[torch.Tensor] = None,
    cu_seqlens_k : Optional[torch.Tensor] = None,
    max_seqlen_q : int = 0,
    max_seqlen_k : int = 0,
) -> torch.Tensor:
    """
    FlashNaylisAttention-2 — API publique.

    Paramètres
    ----------
    q, k, v       : tenseurs d'attention.
                    Mode normal  : [B, H, S, D] / [B, H_KV, S, D]
                    Mode varlen  : [total_tokens, H, D] / [total_tokens, H_KV, D]
    r_q, r_k      : projections relationnelles Naylis.
                    Mode normal  : [B, H_G, S, R]
                    Mode varlen  : [total_tokens, H_G, R]
    graph_scale   : [H_G] — par tête graph, init=0 → transformer classique
    softmax_scale : 1/sqrt(head_dim) si None
    is_causal     : masque causal (toujours True pour l'autorégressif)
    cu_seqlens_*  : pour le sequence packing (varlen)
    max_seqlen_*  : longueur max pour le padding des tiles varlen

    Retourne
    --------
    output : même shape que q
    """
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(q.shape[-1])

    # Vérification des types — le biais graph doit rester en BF16
    assert r_q.dtype == torch.bfloat16, "r_q doit être en bfloat16"
    assert r_k.dtype == torch.bfloat16, "r_k doit être en bfloat16"

    # Vérification du BLOCK_R vs rel_rank
    use_varlen = cu_seqlens_q is not None
    if use_varlen:
        assert r_q.shape[2] == BLOCK_R, f"rel_rank doit être {BLOCK_R}"
    else:
        assert r_q.shape[3] == BLOCK_R, f"rel_rank doit être {BLOCK_R}"

    return FlashNaylisAttnFunc.apply(
        q, k, v, r_q, r_k, graph_scale,
        softmax_scale, is_causal,
        cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION NUMÉRIQUE — à lancer en standalone pour vérifier la correction
# ─────────────────────────────────────────────────────────────────────────────

def _validate_fna2(device: str = "cuda"):
    """
    Compare FNA2 avec la référence SDPA + biais matérialisé.
    Lance : python flash_naylis_attn.py
    """
    import torch
    torch.manual_seed(42)

    B, H, S, D = 2, 8, 128, 64
    H_KV        = 4    # GQA 2:1
    H_G         = 4    # 4 têtes graph + 4 vanilla
    R           = 32   # BLOCK_R

    q  = torch.randn(B, H,    S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k  = torch.randn(B, H_KV, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v  = torch.randn(B, H_KV, S, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    rq = torch.randn(B, H_G,  S, R, device=device, dtype=torch.bfloat16, requires_grad=True)
    rk = torch.randn(B, H_G,  S, R, device=device, dtype=torch.bfloat16, requires_grad=True)
    gs = torch.randn(H_G,           device=device, dtype=torch.float32,  requires_grad=True)

    scale = 1.0 / math.sqrt(D)

    # ── Référence SDPA ────────────────────────────────────────────────────────
    # Expand K/V pour GQA
    gqa = H // H_KV
    k_exp = k.repeat_interleave(gqa, dim=1)   # [B, H, S, D]
    v_exp = v.repeat_interleave(gqa, dim=1)

    # Biais matérialisé en BF16
    rq_r = rq.float()
    rk_r = rk.float()
    bias_full = torch.zeros(B, H, S, S, device=device, dtype=torch.float32)
    bias_full[:, :H_G] = torch.einsum("bhir,bhjr->bhij", rq_r, rk_r) * gs.view(1, H_G, 1, 1)
    bias_full = bias_full.to(torch.bfloat16)

    causal_mask = torch.triu(
        torch.full((S, S), float("-inf"), device=device, dtype=torch.bfloat16), diagonal=1
    )
    attn_mask = bias_full + causal_mask.unsqueeze(0).unsqueeze(0)

    ref_out = torch.nn.functional.scaled_dot_product_attention(
        q, k_exp, v_exp, attn_mask=attn_mask, is_causal=False, scale=scale
    )

    # ── FNA2 ──────────────────────────────────────────────────────────────────
    fna2_out = flash_naylis_attn(q, k, v, rq, rk, gs, softmax_scale=scale, is_causal=True)

    # ── Comparaison ───────────────────────────────────────────────────────────
    max_diff = (ref_out - fna2_out).abs().max().item()
    mean_diff = (ref_out - fna2_out).abs().mean().item()
    print(f"Forward max diff  : {max_diff:.6f}  (tolérance BF16 ≈ 0.01)")
    print(f"Forward mean diff : {mean_diff:.6f}")
    assert max_diff < 0.05, f"ERREUR : diff trop grande ({max_diff:.6f})"
    print("✓ Validation forward réussie")

    # Backward
    dout = torch.randn_like(ref_out)
    ref_out.backward(dout)
    dq_ref, dk_ref, dv_ref = q.grad.clone(), k.grad.clone(), v.grad.clone()
    q.grad = k.grad = v.grad = rq.grad = rk.grad = gs.grad = None

    fna2_out2 = flash_naylis_attn(q, k, v, rq, rk, gs, softmax_scale=scale, is_causal=True)
    fna2_out2.backward(dout)

    print(f"dQ max diff : {(q.grad  - dq_ref).abs().max().item():.6f}")
    print(f"dK max diff : {(k.grad  - dk_ref).abs().max().item():.6f}")
    print(f"dV max diff : {(v.grad  - dv_ref).abs().max().item():.6f}")
    print("✓ Validation backward terminée")


if __name__ == "__main__":
    import sys
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("⚠️  Triton nécessite un GPU CUDA. Validation ignorée.")
        sys.exit(0)
    _validate_fna2(dev)
