# FlashNaylisAttention-2 (FNA2)

> Kernel Triton GPU qui fusionne le biais graphe NaylisAttention dans la boucle de tiling Flash-Attention-2.  
> **Objectif unique** : permettre l'entraînement avec sequence packing + graph_bias simultanément — ce qu'aucune solution existante (SDPA, FA2 varlen) ne permettait.

---

## Le problème résolu

NaylisAttention calcule un biais `B[i,j] = graph_scale × R_q[i] · R_k[j]ᵀ` qui encode la structure de graphe dans l'attention.

Avant FNA2 :

| Scénario | graph_bias | Packing |
|---|---|---|
| SDPA + attn_mask | ✅ appliqué | ❌ impossible |
| FA2 varlen | ❌ ignoré | ✅ fonctionne |
| **FNA2** | ✅ **appliqué** | ✅ **fonctionne** |

De plus, SDPA matérialisait un tenseur `[B, H, S, S]` complet (≈100 MB à S=2048) pour passer le biais via `attn_mask`.

---

## Gain mémoire

Le tenseur `[B, H, S, S]` est **éliminé**. Le biais est calculé tuile par tuile dans les registres GPU et n'est jamais écrit en DRAM.

| Tenseur | SDPA (avant) | FNA2 (après) | Note |
|---|---|---|---|
| Q `[B,H,S,D]` | 25 MB | 25 MB | inchangé |
| K `[B,H_KV,S,D]` | 8 MB | 8 MB | inchangé |
| V `[B,H_KV,S,D]` | 8 MB | 8 MB | inchangé |
| R_q `[B,H_G,S,R]` | 6 MB | 6 MB | inchangé |
| R_k `[B,H_G,S,R]` | 6 MB | 6 MB | inchangé |
| **Biais `[B,H,S,S]`** | **100 MB** | **0 MB** | ← **éliminé** |
| LSE `[B,H,S]` | 0 MB | 0.1 MB | pour le backward |
| Out `[B,H,S,D]` | 25 MB | 25 MB | inchangé |
| **TOTAL** | **~178 MB** | **~78 MB** | **−100 MB / layer** |

*Chiffres à S=2048, B=1, H=12 (220M). En entraînement les gradients doublent le tout.*

Complexité : **O(H × S²) → O(H × S)** pour la mémoire du biais.

---

## Architecture du kernel forward

```
Grid : (⌈S/128⌉, H, B)  ← un programme par (tuile Q, tête, batch)

Programme (pid_m, pid_h, pid_b) :
  Charger Q_i   [128, 64]          depuis DRAM  ← O(S) total
  Charger R_q_i [128, 32]          depuis DRAM  ← O(S) total
  m_i ← -∞,  l_i ← 0,  acc ← 0

  Pour j = 0..⌈S/64⌉ (boucle K) :
    Charger K_j   [64, 64]          depuis DRAM
    Charger R_k_j [64, 32]          depuis DRAM  ← O(64×32) = 4 KB, jamais S×S

    S_ij  ← Q_i @ K_jᵀ × scale               [128, 64]
    B_ij  ← R_q_i @ R_k_jᵀ × graph_scale[h]  [128, 64]  ← FUSION dans registres
    S_ij  ← S_ij + B_ij
    S_ij  ← masque causal + OOB

    m_new ← max(m_i, max(S_ij))
    p     ← exp(S_ij − m_new)
    acc   ← acc × exp(m_i − m_new) + p @ V_j  ← online softmax FA2

  acc ← acc / l_i
  Écrire Out_i [128, 64]  et  LSE_i [128]
```

Le `[B, H, S, S]` n'existe **nulle part**. Chaque programme GPU n'a besoin que de 2 registres [128,32] + [64,32] pour R_q/R_k, soit ~12 KB au lieu de 100 MB.

---

## Fichiers

```
FNA2/
├── flash_naylis_attn.py   Kernels Triton GPU (8 kernels @triton.jit)
│   ├── _fna2_fwd_kernel                  forward standard
│   ├── _fna2_fwd_varlen_kernel           forward sequence packing
│   ├── _fna2_bwd_preprocess_kernel       backward prep (δ = dO·O)
│   ├── _fna2_bwd_dkdv_kernel             backward dK, dV, dR_k
│   ├── _fna2_bwd_dq_kernel               backward dQ, dR_q
│   ├── _fna2_bwd_preprocess_varlen_kernel backward prep varlen
│   ├── _fna2_bwd_dkdv_varlen_kernel      backward dK, dV, dR_k varlen
│   └── _fna2_bwd_dq_varlen_kernel        backward dQ, dR_q varlen
│
├── attention_fna2.py      NaylisAttention avec FNA2 intégré (drop-in replacement)
├── validate_fna2.py       Suite de validation numérique (6 tests)
├── bench_packing.py       Benchmark throughput + VRAM (4 configs comparées)
├── README_FNA2.md         Ce fichier
└── INTEGRATION.md         Guide d'intégration dans le repo NaylisGPT
```

---

## Démarrage rapide

```bash
# 1. Copier dans le repo
cp FNA2/flash_naylis_attn.py  Core/Attention/
cp FNA2/attention_fna2.py     Core/Attention/attention.py   # remplace l'original

# 2. Installer Triton (inclus dans torch 2.x)
pip install triton>=2.2

# 3. Valider
python FNA2/validate_fna2.py --config 60m     # rapide (~30s)
python FNA2/validate_fna2.py --config 220m    # complet

# 4. Benchmarker
python FNA2/bench_packing.py --config 220m    # sweep S=256..4096
python FNA2/bench_packing.py --seq 2048       # S fixé
```

---

## Tests de validation (6 tests)

| Test | Ce qui est vérifié |
|---|---|
| T1 `forward` | FNA2 ≡ SDPA+biais (tolérance BF16) |
| T2 `backward` | dQ, dK, dV, dR_q, dR_k, d_graph_scale corrects |
| T3 `varlen` | Forward packing = 2 passes indépendantes |
| **T3b `varlen_backward`** | **Backward packing = 2 passes indépendantes ← critique** |
| T4 `gs_zero` | FNA2(graph_scale=0) = attention classique (step 0) |
| T5 `memory` | FNA2 utilise moins de VRAM que SDPA+biais |

```bash
python FNA2/validate_fna2.py --test varlen_backward   # le test le plus important
```

---

## Têtes supportées

| Type | Support | Comportement |
|---|---|---|
| Graph asymétriques | ✅ | `B[i,j] = gs × R_q_i · R_k_jᵀ` (≠ `B[j,i]`) |
| Vanilla | ✅ | Pas de chargement R_q/R_k — identique à FA2 pur |
| Graph symétriques | ❌ | Éliminés après Run 5, incompatibles |

GQA (Grouped Query Attention) est géré nativement dans le kernel via le ratio `H // H_KV`.

---

## Garanties importantes

**BF16 préservé** — R_q et R_k sont toujours castés en BF16 avant le kernel. graph_scale reste en FP32 (AdamW). TE/FP4 ne peut pas intercepter le biais puisqu'il n'existe jamais comme tenseur.

**Step 0 identique** — `graph_scale` initialisé à 0 → biais nul → FNA2 = attention classique exacte. Reproductibilité garantie.

**Isolation séquences** — Le masque causal varlen est relatif à chaque séquence (`offs_m >= offs_n` dans les positions locales). Aucune fuite cross-séquence possible.

---

## Limites connues

1. **`BLOCK_R = 32` fixé** — doit correspondre à `rel_rank`. Si vous changez `rel_rank`, modifier `BLOCK_R` dans `flash_naylis_attn.py` ligne ~45.
2. **head_dim : 64 ou 128 seulement** — contrainte Triton `tl.dot` (puissance de 2, ≥ 16).
3. **Têtes symétriques non supportées** — cohérent avec la direction du projet post-Run 5.
4. **`soft_cap`** — désactive FNA2 (fallback SDPA automatique).

---

## Roadmap

| Phase | Statut | Description |
|---|---|---|
| P0 | ✅ DONE | TE NVFP4 intégration (28/28 tests SM120) |
| P1 | ✅ DONE | FNA2 standard + varlen forward + backward |
| P2 | ⏳ Next | Kernel FNA2 NVFP4 — biais en FP4 natif tile-par-tile |
| P3 | ⏳ | Run 220M avec FNA2 + sequence packing activé |

---

## Référence algorithme

FNA2 implémente l'algorithme Flash-Attention-2 (Dao 2023) avec une modification dans la boucle K :

```
// FA2 standard :
S_ij = Q_i @ K_j^T * scale

// FNA2 (modification) :
S_ij = Q_i @ K_j^T * scale
if graph_head:
    B_ij = gs_h * (R_q_i @ R_k_j^T)   // tile [128,64] en registres
    S_ij = S_ij + B_ij
```

Le backward utilise la recomputation (on stocke LSE, pas la matrice d'attention) — même technique que FA2.
