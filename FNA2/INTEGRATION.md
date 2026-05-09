# Guide d'intégration FNA2 dans NaylisGPT

Ce guide explique comment intégrer FNA2 dans l'architecture NaylisGPT existante (`Core/Attention/attention.py` + `Core/Model/HessGpt.py`), et comment modifier le loop d'entraînement pour activer le sequence packing.

---

## Vue d'ensemble

```
Avant FNA2                          Après FNA2
──────────────────────────────      ──────────────────────────────
attention.py                        attention.py  ← remplacé par attention_fna2.py
  _compute_graph_bias()               [pas de _compute_graph_bias() en FNA2]
  → matérialise [B,H,S,S]            → biais calculé tile-par-tile dans kernel

HessGpt.py                          HessGpt.py
  from attention import ...           from attention import ...  ← même ligne
  NaylisAttention(...)                NaylisAttention(...)  ← même signature

pretrain_60M.py / pretrain_220M.py  pretrain_*.py
  # batch de séquences paddées        # batch packé avec cu_seqlens
  model(x, mask=pad_mask)             model(x_packed,
                                            cu_seqlens_q=cu_q,
                                            cu_seqlens_k=cu_k,
                                            max_seqlen_q=max_s,
                                            max_seqlen_k=max_s)
```

---

## Étape 1 — Copier les fichiers

```bash
# Depuis la racine du repo NaylisGPT
cp FNA2/flash_naylis_attn.py   Core/Attention/flash_naylis_attn.py
cp FNA2/attention_fna2.py      Core/Attention/attention.py        # écrase l'original
```

`attention_fna2.py` est un **drop-in replacement** — la classe `NaylisAttention` a exactement la même signature `__init__` et `forward`. `HessGpt.py` n'a pas besoin d'être modifié.

> **Important** : les deux fichiers doivent être dans le **même dossier** (`Core/Attention/`). `attention_fna2.py` importe `flash_naylis_attn` par chemin relatif.

---

## Étape 2 — Vérifier `rel_rank = 32`

FNA2 est compilé avec `BLOCK_R = 32`. Votre config actuelle :

```python
# HessGpt.py
rel_rank = 32   # ← déjà correct pour les configs 60M et 220M
```

Si vous changez `rel_rank`, modifier aussi `BLOCK_R` dans `flash_naylis_attn.py` :

```python
# flash_naylis_attn.py — ligne ~45
BLOCK_R = 32   # ← changer ici aussi
BLOCK_M = 128
BLOCK_N = 64
```

---

## Étape 3 — Valider avant d'entraîner

```bash
# Validation numérique complète (≈ 2 min sur RTX Pro 6000)
python FNA2/validate_fna2.py --config 220m

# Test le plus critique (backward varlen = backward packing)
python FNA2/validate_fna2.py --test varlen_backward --config 220m

# Benchmark mémoire + throughput
python FNA2/bench_packing.py --config 220m --seq 2048
```

Si tous les tests sont verts, FNA2 est prêt.

---

## Étape 4 — Modifier le loop d'entraînement pour le packing

C'est la seule vraie modification de code dans `pretrain_*.py`. Il faut construire les `cu_seqlens` à partir du batch.

### 4a — Sans le DataLoader (modification minimale)

```python
# pretrain_60M.py / pretrain_220M.py
# ─────────────────────────────────────────────────────────────────────
# AVANT (séquences paddées, pas de packing)
# ─────────────────────────────────────────────────────────────────────
for x, y in dataloader:
    x = x.to(device)           # [B, S]
    y = y.to(device)           # [B, S]

    logits, _ = model(x)       # NaylisAttention reçoit [B, S, embed]
    loss = cross_entropy(logits, y)
    loss.backward()
    optimizer.step()

# ─────────────────────────────────────────────────────────────────────
# APRÈS (sequence packing avec FNA2)
# ─────────────────────────────────────────────────────────────────────
for x_list, y_list in packed_dataloader:
    # x_list : liste de B séquences de longueurs variables [s1, s2, ...]
    # y_list : idem pour les labels

    # Construire le batch packé
    x_packed  = torch.cat(x_list, dim=0).to(device)   # [total_tokens]
    y_packed  = torch.cat(y_list, dim=0).to(device)   # [total_tokens]
    seqlens   = torch.tensor([s.shape[0] for s in x_list], dtype=torch.int32)
    cu_seqlens = torch.zeros(len(x_list) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens, 0).to(device)
    max_seqlen = int(seqlens.max().item())

    # Forward avec packing
    logits, _ = model(
        x_packed,                      # [total_tokens, embed_dim]
        cu_seqlens_q = cu_seqlens,
        cu_seqlens_k = cu_seqlens,
        max_seqlen_q = max_seqlen,
        max_seqlen_k = max_seqlen,
    )
    loss = cross_entropy(logits, y_packed)
    loss.backward()
    optimizer.step()
```

### 4b — Fonction helper `pack_batch`

```python
def pack_batch(sequences: list[torch.Tensor], device):
    """
    Transforme une liste de séquences de longueurs variables en batch packé.

    Args:
        sequences : liste de tenseurs 1D [s_i] — token IDs
        device    : torch.device

    Returns:
        x_packed   : [total_tokens]        — tokens concaténés
        cu_seqlens : [N+1]  int32          — offsets cumulatifs
        max_seqlen : int                   — longueur max (pour le kernel)
    """
    x_packed   = torch.cat(sequences, dim=0).to(device)
    seqlens    = torch.tensor([s.shape[0] for s in sequences], dtype=torch.int32)
    cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens, 0).to(device)
    max_seqlen = int(seqlens.max().item())
    return x_packed, cu_seqlens, max_seqlen


# Utilisation dans le loop :
for batch in dataloader:
    x_packed, cu_q, max_s = pack_batch(batch["input_ids"], device)
    y_packed, _,    _     = pack_batch(batch["labels"],    device)

    logits, _ = model(
        x_packed,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_q,
        max_seqlen_q=max_s, max_seqlen_k=max_s,
    )
    loss = F.cross_entropy(logits.view(-1, vocab_size), y_packed.view(-1))
    loss.backward()
```

---

## Étape 5 — Adapter le DataLoader pour le packing

Le packing est plus efficace si les séquences sont regroupées par longueur similaire (évite de gaspiller des tokens de padding).

### Option simple : concat-then-chunk

```python
class PackedDataset(torch.utils.data.Dataset):
    """
    Concatène tous les documents et les découpe en chunks de longueur max_seq_len.
    Chaque chunk est une séquence — pas de document boundary.
    Utilisé avec collate_packed() ci-dessous.
    """

    def __init__(self, token_ids: torch.Tensor, max_seq_len: int):
        self.data        = token_ids
        self.max_seq_len = max_seq_len
        self.n_chunks    = len(token_ids) // max_seq_len

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        start = idx * self.max_seq_len
        x = self.data[start : start + self.max_seq_len]
        y = self.data[start + 1 : start + self.max_seq_len + 1]
        return x, y


def collate_packed(batch, device):
    """
    Reçoit une liste de (x, y) de même longueur → batch packé.
    Pour des séquences de longueur identique, cu_seqlens est trivial.
    """
    xs, ys = zip(*batch)
    x_packed   = torch.stack(xs).view(-1)   # [B * S]
    y_packed   = torch.stack(ys).view(-1)

    S          = xs[0].shape[0]
    B          = len(xs)
    seqlens    = torch.full((B,), S, dtype=torch.int32)
    cu_seqlens = torch.zeros(B + 1, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seqlens, 0).to(device)

    return (x_packed.to(device), y_packed.to(device),
            cu_seqlens, cu_seqlens.clone(), S, S)


# Utilisation :
dataset = PackedDataset(all_token_ids, max_seq_len=2048)
loader  = DataLoader(dataset, batch_size=4,
                     collate_fn=lambda b: collate_packed(b, device))

for x_p, y_p, cu_q, cu_k, msl_q, msl_k in loader:
    logits, _ = model(x_p, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
                      max_seqlen_q=msl_q, max_seqlen_k=msl_k)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y_p.view(-1))
    loss.backward()
    ...
```

### Option avancée : document-aware packing

Pour respecter les boundaries de documents (éviter que la fin d'un doc soit vue par le début du suivant) :

```python
def pack_documents(docs: list[list[int]], max_seq_len: int, eos_id: int = 2):
    """
    Emballage greedy de documents dans des bins de max_seq_len tokens.
    Ajoute un token EOS entre documents. Retourne des bins prêts pour FNA2.

    Chaque bin = une séquence dans le batch packé (une entrée de cu_seqlens).
    """
    bins, current_bin = [], []

    for doc in docs:
        doc_tokens = doc + [eos_id]
        if len(current_bin) + len(doc_tokens) > max_seq_len:
            if current_bin:
                bins.append(current_bin)
            current_bin = doc_tokens[:max_seq_len]
        else:
            current_bin.extend(doc_tokens)

    if current_bin:
        bins.append(current_bin)

    return bins
```

---

## Étape 6 — Modifier `HessGpt.forward` pour propager `cu_seqlens`

`HessGpt.py` appelle `NaylisAttention.forward` dans chaque couche. Il faut propager les `cu_seqlens` jusqu'à chaque couche :

```python
# Core/Model/HessGpt.py

class HessGpt(nn.Module):
    ...
    def forward(
        self,
        x,
        mask=None,
        past_kv=None,
        use_kv_cache=False,
        cu_seqlens_q=None,     # ← ajouter ces 4 paramètres
        cu_seqlens_k=None,
        max_seqlen_q=None,
        max_seqlen_k=None,
    ):
        # Mode varlen : x est [total_tokens] (IDs), pas [B, S]
        is_varlen = (cu_seqlens_q is not None)

        if is_varlen:
            x = self.token_embeddings(x)   # [total_tokens, embed_dim]
        else:
            B, S = x.shape
            x = self.token_embeddings(x)   # [B, S, embed_dim]

        new_kv_caches = []
        for i, block in enumerate(self.blocks):
            past  = past_kv[i] if past_kv is not None else None
            x, kv = block(
                x,
                mask         = mask,
                past_kv      = past,
                use_kv_cache = use_kv_cache,
                cu_seqlens_q = cu_seqlens_q,
                cu_seqlens_k = cu_seqlens_k,
                max_seqlen_q = max_seqlen_q,
                max_seqlen_k = max_seqlen_k,
            )
            new_kv_caches.append(kv)

        x      = self.ln_final(x)
        logits = self.output_head(x)
        return logits, new_kv_caches if use_kv_cache else None
```

Vérifier aussi `TransformerBlock.forward` pour qu'il accepte et propage les mêmes paramètres vers `NaylisAttention`.

---

## Étape 7 — Vérification finale en entraînement

```python
# Sanity check rapide avant un vrai run
import torch
from Core.Model.HessGpt import HessGpt

model = HessGpt(...).to("cuda").to(torch.bfloat16)

# 4 séquences de 512 tokens packées → 2048 tokens total
S, B  = 512, 4
tokens = torch.randint(0, 32000, (B * S,), device="cuda")
seqlens = torch.tensor([S] * B, dtype=torch.int32)
cu_q    = torch.zeros(B + 1, dtype=torch.int32, device="cuda")
cu_q[1:] = torch.cumsum(seqlens, 0)

logits, _ = model(tokens, cu_seqlens_q=cu_q, cu_seqlens_k=cu_q,
                  max_seqlen_q=S, max_seqlen_k=S)
loss = logits.float().mean()
loss.backward()
print("✅ Forward + backward packing OK")
print(f"   logits shape : {logits.shape}")   # [B*S, vocab_size]
```

---

## Ce qui NE change PAS

- Tous les hyperparamètres de `NaylisAttention` (`embed_dim`, `num_heads`, `n_kv_heads`, `rel_rank`, `graph_heads`, `vanilla_heads`, etc.)
- Les poids du modèle (aucun nouveau paramètre ajouté)
- Le comportement à `graph_scale=0` (step 0 = transformer classique)
- La génération/inférence (pas de packing en inférence, même code qu'avant)
- La sauvegarde et le chargement des checkpoints

---

## Détection automatique (sans changement de code)

`attention_fna2.py` détecte FNA2 au démarrage et affiche :

```
  ⚡ FNA2 (FlashNaylisAttention-2) — Triton — Blackwell SM120
```

Si Triton n'est pas installé ou si `rel_rank ≠ 32`, il affiche :

```
  ⚠️  FNA2 non disponible (...), fallback SDPA
```

Et continue à fonctionner normalement avec SDPA (sans packing, avec biais matérialisé) — aucune erreur.

---

## Résumé des fichiers modifiés

| Fichier | Modification | Obligatoire |
|---|---|---|
| `Core/Attention/attention.py` | Remplacé par `attention_fna2.py` | ✅ |
| `Core/Attention/flash_naylis_attn.py` | Copié (nouveau fichier) | ✅ |
| `Core/Model/HessGpt.py` | Propager `cu_seqlens_*` dans `forward` | ✅ si packing |
| `pretrain_60M.py` / `pretrain_220M.py` | Loop d'entraînement avec `pack_batch` | ✅ si packing |
| `Core/Attention/TransformerBlock` | Propager `cu_seqlens_*` vers `NaylisAttention` | ✅ si packing |

La modification minimale (attention uniquement, sans packing) ne nécessite que les 2 premières lignes.
