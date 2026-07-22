# Online Query Pipeline — Comment une question traverse le système

> Ce document décrit **en détail** ce qui se passe depuis le moment où l'utilisateur soumet une question jusqu'à la réponse générée. Focus particulier sur le **Hybrid Retrieval** et son rôle central.

---

## Table des matières

1. [Vue macro du pipeline](#1-vue-macro-du-pipeline)
2. [Étape 0 — Cache sémantique (QueryCache)](#2-étape-0--cache-sémantique-querycache)
3. [Étape 1 — Embedding de la requête](#3-étape-1--embedding-de-la-requête)
4. [Étape 2 — Hybrid Retrieval (FAISS + BM25 + RRF)](#4-étape-2--hybrid-retrieval-faiss--bm25--rrf)
   - 4.1 [Qu'est-ce que le Hybrid Retrieval ?](#41-quest-ce-que-le-hybrid-retrieval-)
   - 4.2 [Bras 1 — Dense Retrieval (FAISS)](#42-bras-1--dense-retrieval-faiss)
   - 4.3 [Bras 2 — Sparse Retrieval (BM25)](#43-bras-2--sparse-retrieval-bm25)
   - 4.4 [Fusion RRF — Reciprocal Rank Fusion](#44-fusion-rrf--reciprocal-rank-fusion)
   - 4.5 [Routing par langue](#45-routing-par-langue)
   - 4.6 [Filtrage post-fusion](#46-filtrage-post-fusion)
5. [Étape 3 — Re-ranking (Cross-Encoder)](#5-étape-3--re-ranking-cross-encoder)
6. [Étape 4 — GraphRAG (optionnel)](#6-étape-4--graphrag-optionnel)
7. [Étape 5 — Vision / VQA (optionnel)](#7-étape-5--vision--vqa-optionnel)
8. [Étape 6 — Génération de la réponse (LLM)](#8-étape-6--génération-de-la-réponse-llm)
9. [Étape 7 — Citations et post-processing](#9-étape-7--citations-et-post-processing)
10. [Étape 8 — Sauvegarde dans le cache](#10-étape-8--sauvegarde-dans-le-cache)
11. [Mode streaming SSE](#11-mode-streaming-sse)
12. [Pourquoi Hybrid et pas seulement Dense ?](#12-pourquoi-hybrid-et-pas-seulement-dense-)
13. [Paramètres de performance et tuning](#13-paramètres-de-performance-et-tuning)
14. [Flux complet en pseudo-code](#14-flux-complet-en-pseudo-code)

---

## 1. Vue macro du pipeline

```
Question utilisateur
        │
        ▼
┌───────────────────────────┐
│  BUG-7: Clarification     │  ← si confirmed=False et question ambiguë
│  (JSON-mode, 150 tokens)  │     → retourne ClarificationResponse, s'arrête ici
└───────────────────────────┘
        │
        ▼
┌───────────────────────────┐
│  QueryCache.lookup()      │  ← cosine similarity sur SQLite
│  seuil : 0.85             │     → HIT  = réponse instantanée, fin
└───────────────┬───────────┘     → MISS = continue
                │
                ▼
┌───────────────────────────┐
│  EmbeddingGenerator       │
│  .embed_query(question)   │  ← BGE-M3 dense 1024d
│                           │     + HyDE optionnel (texte hypothétique)
│  → QueryEmbedding         │     + Nomic 256d optionnel
└───────────────┬───────────┘     + ColBERT token-vecs optionnel
                │
                ▼
┌───────────────────────────────────────────────────────────┐
│           HYBRID RETRIEVAL  (VideoStore.search)           │
│                                                           │
│   BGE-M3 dense vector ──► FAISS HNSW  ──┐                │
│                                          ├─► RRF Fusion   │
│   BM25 tokenized query ──► BM25Okapi  ──┘  (weighted)    │
│                                                           │
│   Routing par langue: AR / EN / FR / *                    │
│   Filtres: video_id, speaker, chunk_type, filter_tags     │
│                                                           │
│   → top-20 candidats avec (dense_score, sparse_score,    │
│     fused_score, strategy)                               │
└───────────────┬───────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────┐
│  CrossEncoderReranker     │
│  .rerank(query, results)  │  ← BAAI/bge-reranker-v2-m3
│  → top-k résultats        │     score cross-attention query↔chunk
│    avec rerank_score      │
└───────────────┬───────────┘
                │
                ├──────────────────┐
                ▼                  ▼
┌─────────────────────┐  ┌────────────────────────┐
│  GraphSearch.ask()  │  │  VQAEngine.answer()    │
│  (si graph_enabled) │  │  (si vision=True)      │
│  → graph context    │  │  → visual evidence     │
└──────────┬──────────┘  └──────────┬─────────────┘
           │                        │
           └──────────┬─────────────┘
                      ▼
┌───────────────────────────────────────┐
│  LLMClient.synthesize_answer()        │
│  ou .stream_answer()                  │
│                                       │
│  Inputs:                              │
│   - question                          │
│   - video_metadata (titre, résumé...) │
│   - top-k chunks reranked             │
│   - graph context (optionnel)         │
│   - visual evidence (optionnel)       │
│   - intent + language hint            │
│                                       │
│  → LLMAnswer (answer + citations)     │
└───────────────┬───────────────────────┘
                │
                ▼
┌───────────────────────────┐
│  QueryCache.save()        │
│  → SQLite (question_emb   │
│    + response_json)       │
└───────────────────────────┘
                │
                ▼
       Réponse JSON / SSE
```

---

## 2. Étape 0 — Cache sémantique (QueryCache)

**Fichier** : `core/query_cache.py`

Avant tout traitement coûteux, le système vérifie si une réponse similaire existe déjà en cache.

### Comment ça marche

```
question → embed_query() → vecteur dense (1024d)
                │
                ▼
    SELECT * FROM query_cache WHERE video_id = ?
                │
                ▼
    Pour chaque entrée en cache :
        cosine(question_emb, stored_emb) >= seuil ?
                │
         OUI ──►│──► retourner response_json (hit)
                │
         NON ──►│──► continuer le pipeline (miss)
```

### Schéma SQLite

```sql
CREATE TABLE query_cache (
    id           INTEGER PRIMARY KEY,
    video_id     TEXT,
    question     TEXT,           -- texte original
    question_emb BLOB,           -- float32 numpy array (1024d BGE-M3)
    response_json TEXT,          -- JSON complet de /ask
    hit_count    INTEGER,        -- combien de fois ce cache a été utilisé
    created_at   REAL,           -- timestamp unix
    last_hit_at  REAL
);
```

### Paramètres

| Paramètre | Valeur | Effet |
|-----------|--------|-------|
| `query_cache_similarity` | 0.85 | Score cosine minimum pour un hit |
| `query_cache_max_entries` | 10 000 | Nb max d'entrées (LRU au-delà) |
| `query_cache_ttl_days` | 30 | Durée de vie d'une entrée |
| `use_cache` (par requête) | `True` | Peut être désactivé per-request |

### Pourquoi ce seuil à 0.85 ?

Un score cosine de 0.85 signifie que les vecteurs BGE-M3 sont très proches. En pratique :
- "Qu'est-ce que le protocole BB84 ?" → "Explique le protocole BB84" → ~0.91 → **HIT**
- "Qu'est-ce que BB84 ?" → "Qui a inventé BB84 ?" → ~0.78 → **MISS** (intentions différentes)

---

## 3. Étape 1 — Embedding de la requête

**Fichier** : `core/embeddings.py` — classe `EmbeddingGenerator`

### Ce qui se passe

```python
query_emb = embedder.embed_query(question)
```

Cette méthode produit un objet `QueryEmbedding` :

```python
@dataclass
class QueryEmbedding:
    text: str                                # question originale
    bge_dense: List[float]                   # 1024d — vecteur principal
    bge_sparse: Optional[SparseVectorData]   # {token_id: weight} — lexical
    nomic_dense: Optional[List[float]]       # 256d — matryoshka (optionnel)
    colbert_query: Optional[List[List[float]]] # [token_count, 128d] (optionnel)
    expanded_text: Optional[str]             # HyDE : texte hypothétique généré
```

### BGE-M3 : le cœur de l'embedding

BGE-M3 (BAAI) est un modèle multi-vector exceptionnel car il produit **en un seul passage** :

```
Texte de la question
        │
        ▼
    BGE-M3 encoder (FlagEmbedding)
        │
        ├──► dense_vecs    [1024d float32]  ← représentation sémantique globale
        ├──► lexical_weights {token: weight} ← représentation sparse lexicale
        └──► colbert_vecs  [N_tokens × 128d] ← vecteurs par token
```

**Pourquoi 1024d ?** BGE-M3 utilise une architecture basée sur XLM-RoBERTa étendue. Les 1024 dimensions permettent une représentation sémantique très riche, capable de capturer des nuances en arabe, anglais et français.

### HyDE — Hypothetical Document Embedding (optionnel)

Si `hyde_enabled=True` et la question est courte (≤ 15 mots), le système génère un **texte hypothétique** qui ressemble à ce que dirait la vidéo en réponse :

```
Question: "BB84 protocol"
                │
                ▼ (Ollama, 100 tokens, temp=0.3)
"BB84 is a quantum key distribution protocol proposed by Bennett
and Brassard in 1984. It uses the polarization of photons to
establish a secret key between two parties..."
                │
                ▼
expanded_text = "BB84 protocol BB84 is a quantum key distribution..."
```

Le vecteur dense est calculé sur `question + expanded_text`. Cela booste le rappel pour les questions courtes ou ambiguës car le vecteur est plus proche du vocabulaire utilisé dans la transcription.

### Cache des embeddings

```python
# Avant d'appeler BGE-M3 :
cached = embedding_cache.get(question_text)
if cached:
    return cached   # 30 jours de TTL — BGE-M3 est déterministe

# Après l'appel :
embedding_cache.put(question_text, result)
```

Le cache est persisté sur disque (`data/faiss_store/embedding_cache.pkl`). Un appel à BGE-M3 prend ~80-200ms sur CPU ; le cache le réduit à ~0.1ms.

---

## 4. Étape 2 — Hybrid Retrieval (FAISS + BM25 + RRF)

**Fichier** : `core/storage.py` — méthode `VideoStore.search()`

C'est **l'étape la plus importante** du pipeline. Voici tout ce qui s'y passe.

### 4.1 Qu'est-ce que le Hybrid Retrieval ?

Le Hybrid Retrieval combine **deux paradigmes de recherche fondamentalement différents** :

| | Dense Retrieval (FAISS) | Sparse Retrieval (BM25) |
|--|------------------------|------------------------|
| **Représentation** | Vecteur flottant 1024d | Sac de mots pondérés |
| **Similarité** | Cosine / produit scalaire | Score TF-IDF (BM25) |
| **Force** | Comprend le sens, les synonymes, les reformulations | Précis sur les mots exacts, noms propres, termes techniques |
| **Faiblesse** | Peut rater les correspondances exactes (entités, acronymes) | Aveugle à la sémantique, aux synonymes |
| **Exemple fort** | "cryptographie quantique" → trouve "chiffrement par photons" | "BB84" → trouve exactement "BB84" |
| **Exemple faible** | "BB84" peut être noyé dans le bruit sémantique | "chiffrement quantique" rate un chunk qui dit "protocole QKD" |

**Le Hybrid est supérieur aux deux** car il capture à la fois le sens ET les mots exacts.

### 4.2 Bras 1 — Dense Retrieval (FAISS)

**Qu'est-ce que FAISS ?**

FAISS (Facebook AI Similarity Search) est une bibliothèque C++ permettant de rechercher très rapidement le vecteur le plus proche parmi des millions de vecteurs. Il remplace une comparaison naïve O(N) par une structure d'index approximative.

**Index utilisé : HNSW (Hierarchical Navigable Small World)**

```
Index HNSW — structure multi-couches
    Couche 2 (sparse)  : nœuds bien connectés, vue globale
    Couche 1 (medium)  : connexions moyennes
    Couche 0 (dense)   : tous les vecteurs, connexions locales

Recherche : entre par la couche haute, navigue vers le plus proche,
            descend progressivement. O(log N) au lieu de O(N).
```

**Paramètres HNSW dans ce projet** :
```python
faiss_hnsw_m = 32                # connexions par nœud — plus = meilleur rappel, plus de RAM
faiss_hnsw_ef_construction = 200 # qualité de la construction de l'index
faiss_hnsw_ef_search = 128       # candidats explorés à la recherche
```

**Index par langue** : le VideoStore maintient **4 index FAISS séparés** :
- `_faiss["ar"]` — chunks en arabe
- `_faiss["en"]` — chunks en anglais
- `_faiss["fr"]` — chunks en français
- `_faiss["*"]` — tous les chunks (fallback cross-langue)

```python
# Recherche dense :
lang_key = self._route_lang(query_lang)   # ex: "ar"

# 1. Index langue exacte
dense_results = self._faiss[lang_key].search(qvec, candidate_k)

# 2. Fallback cross-langue (ajoute les chunks non déjà trouvés)
fallback_dense = self._faiss["*"].search(qvec, candidate_k)
seen_dense = set(cid for cid, _ in dense_results)
for cid, score in fallback_dense:
    if cid not in seen_dense:
        dense_results.append((cid, score))
```

**Pourquoi des index séparés par langue ?** Si la vidéo est en arabe et que la question est en arabe, chercher dans l'index arabe évite que des chunks anglais (même vectoriellement proches) ne viennent polluer les résultats. La précision est nettement meilleure.

**Normalisation L2** : avant chaque insertion ou recherche dans FAISS, les vecteurs sont L2-normalisés. Cela transforme le produit scalaire interne en similarité cosine. Les vecteurs BGE-M3 sont déjà normalisés mais FAISS renormalise quand même pour garantir la cohérence.

```python
vec = vector.reshape(1, -1).astype(np.float32)
faiss.normalize_L2(vec)   # ← produit scalaire = cosine après normalisation
```

**Résultat** : liste de `(chunk_id, cosine_score)` triée par score décroissant.

### 4.3 Bras 2 — Sparse Retrieval (BM25)

**Qu'est-ce que BM25 ?**

BM25 (Best Match 25) est une fonction de scoring **statistique** qui mesure la pertinence d'un document par rapport à une requête en se basant sur la fréquence des termes et leur rareté dans le corpus.

**Formule BM25 pour un document D et une requête Q** :

```
BM25(D, Q) = Σ  IDF(qi) × [f(qi, D) × (k1 + 1)] / [f(qi, D) + k1 × (1 - b + b × |D| / avgdl)]
           qi ∈ Q

où :
  f(qi, D) = fréquence du terme qi dans D
  |D|       = longueur de D en tokens
  avgdl     = longueur moyenne des documents du corpus
  k1 = 1.5  (saturation TF — récompense les occurrences multiples, mais avec diminution)
  b  = 0.75 (normalisation longueur — pénalise les docs longs)
  IDF(qi)   = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
            = log rareté du terme (rare = plus de poids)
```

**En pratique dans ce projet** :

```python
# Tokenisation language-aware
def tokenize(text, language):
    if language in ("ar", "fa", "ur"):
        # Normalisation arabe :
        # - Suppression des diacritiques (حركات)
        # - Suppression du tatweel (ـ)
        # - Normalisation des hamzas (أإآ → ا)
        # - Normalisation du ta'marbouta (ة → ه)
        text = remove_diacritics(text)
        ...
    tokens = re.split(r"[\s.,;:!?...]+", text.lower())
    return [t for t in tokens if len(t) >= 2]

# BM25 par langue
bm25_key = "ar"  # pour une requête arabe
sparse_results = self._bm25["ar"].search(query_text, candidate_k)
```

**Corpus BM25 par langue** :
```python
self._bm25 = {
    "ar": LanguageBM25Index("ar"),  # corpus tokenisé arabe
    "en": LanguageBM25Index("en"),  # corpus tokenisé anglais
    "fr": LanguageBM25Index("fr"),  # corpus tokenisé français
    "*":  LanguageBM25Index("en"),  # fallback cross-langue
}
```

**Rebuild lazy** : `BM25Okapi` ne supporte pas les mises à jour incrémentales. Chaque nouveau chunk est accumulé (`add_many()`), et le rebuild complet n'a lieu qu'au moment de la **première recherche** après une modification, ou via `prewarm()` explicite à la fin de l'ingestion.

**Résultat** : liste de `(chunk_id, bm25_score)` triée par score décroissant.

### 4.4 Fusion RRF — Reciprocal Rank Fusion

**Problème** : les scores FAISS (cosine, entre -1 et 1) et BM25 (arbitraire, peut dépasser 10) ne sont **pas comparables**. On ne peut pas additionner cosine_score + bm25_score directement.

**Solution : RRF (Reciprocal Rank Fusion)**

RRF ignore les valeurs absolues des scores et se base uniquement sur le **rang** de chaque résultat dans chaque liste. La formule :

```
RRF_score(d) = Σ  w_channel × 1 / (k + rank_channel(d))
             channel

où :
  k = 60       (constante de régularisation — atténue l'impact du rang 1)
  rank(d)      = position du document dans la liste triée (1-indexed)
  w_channel    = poids du canal (dense_w ou sparse_w, selon la langue)
```

**Implémentation dans `_rrf_fuse()`** :

```python
# Contribution du canal dense
for rank, (cid, raw_cosine_score) in enumerate(dense_results):
    scores[cid]["fused"] += dense_w / (rrf_k + rank + 1)
    scores[cid]["dense"]  = float(raw_cosine_score)   # stocké pour affichage
    dense_set.add(cid)

# Contribution du canal sparse
for rank, (cid, raw_bm25_score) in enumerate(sparse_results):
    scores[cid]["fused"] += sparse_w / (rrf_k + rank + 1)
    scores[cid]["sparse"] = float(raw_bm25_score)
    sparse_set.add(cid)

# Tri final par fused_score
ranked = sorted(scores.items(), key=lambda x: x[1]["fused"], reverse=True)
```

**Exemple concret** avec k=60 :

| Chunk | Rang FAISS | Rang BM25 | RRF Dense (w=0.65) | RRF Sparse (w=0.35) | RRF Total |
|-------|-----------|-----------|-------------------|--------------------|-----------|
| C1 | 1 | 5 | 0.65/61 = **0.01066** | 0.35/65 = 0.00538 | **0.01604** |
| C2 | 3 | 1 | 0.65/63 = 0.01032 | 0.35/61 = **0.00574** | 0.01606 |
| C3 | 2 | non-présent | 0.65/62 = 0.01048 | 0 | 0.01048 |
| C4 | non-présent | 2 | 0 | 0.35/62 = 0.00565 | 0.00565 |

**Interprétation** :
- C2 gagne légèrement sur C1 car il est #1 en BM25 (terme exact trouvé)
- C3 (dense seulement, rang 2) bat C4 (sparse seulement, rang 2) grâce au poids dense plus élevé en anglais
- Un chunk absent d'une liste reçoit 0 pour ce canal

**Stratégie attribuée** :
```python
if cid in dense_set and cid in sparse_set:
    strategy = "hybrid_rrf"    # trouvé par les deux — plus fiable
elif cid in dense_set:
    strategy = "dense_only"
elif cid in sparse_set:
    strategy = "sparse_only"
```

### 4.5 Routing par langue

**C'est ici que le projet prend en compte la spécificité de l'arabe.**

```python
# Poids de fusion adaptatifs par langue
AR : dense_w=0.35, sparse_w=0.65  # arabe → BM25 plus précis sur la morphologie
EN : dense_w=0.65, sparse_w=0.35  # anglais → dense meilleur
FR : dense_w=0.50, sparse_w=0.50  # français → équilibré
```

**Pourquoi l'arabe a-t-il besoin de plus de BM25 ?**

L'arabe a une morphologie **agglutinante** très riche. Un même radical peut produire des dizaines de formes selon les préfixes/suffixes (articles, pronoms, cas...). Par exemple :

```
كتب  (kataba = il écrit)
يكتب (yaktub = il écrira)
كاتب (katib = écrivain)
مكتوب (maktub = écrit/lettre)
مكتبة (maktaba = bibliothèque)
```

Le tokenizer arabe dans ce projet normalise ces formes avant BM25 :
- Suppression des diacritiques : `كَتَبَ` → `كتب`
- Normalisation hamzas : `أحمد`/`إحمد`/`آحمد` → `احمد`
- Normalisation ta'marbouta : `مكتبة` → `مكتبه`

Cette normalisation rend BM25 particulièrement efficace pour l'arabe : les variantes morphologiques convergent vers la même forme tokenisée.

**Détection de la langue de la requête** :

```python
query_lang = language or detect_language(query_text)
lang_key   = self._route_lang(query_lang)  # "ar", "en", "fr" ou "*"
```

La détection utilise :
1. `lingua` (si installé) — haute précision
2. Heuristique Unicode : ratio de caractères arabes `[؀-ۿ]` vs latins
3. Phrases françaises spécifiques multi-mots (jamais un seul mot pour éviter les faux positifs)

### 4.6 Filtrage post-fusion

Après fusion RRF, un filtre `_passes(cid)` est appliqué sur chaque résultat :

```python
def _passes(cid: str) -> bool:
    meta = self._meta.get(cid, {})

    # Filtrer par vidéo spécifique
    if video_id and meta.get("video_id") != video_id:
        return False

    # Filtrer par locuteur
    if speaker and meta.get("speaker") != speaker:
        return False

    # Exclure les chunks visuels si vision=False
    if not with_vision and meta.get("chunk_type") == "visual":
        return False

    # Filtrer par type de chunk
    if chunk_type and meta.get("chunk_type") != chunk_type:
        return False

    # Filtrer par tags personnalisés
    if filter_tags:
        chunk_tags = meta.get("metadata", {}).get("filter_tags", [])
        if not any(tag in chunk_tags for tag in filter_tags):
            return False

    return True
```

**Score cache** : les résultats sont mis en cache dans `ScoreCache` (LRU, TTL 30 min) :
```python
cached = self._cache.get(query_text, video_id)
if cached is not None:
    return cached[:effective_limit]   # hit instant
```

**Pool de candidats** : le pipeline récupère `candidate_k = max(top_k × 5, 100)` candidats dans FAISS et BM25 avant la fusion, pour s'assurer que le reranker a un pool suffisant même si certains sont filtrés.

---

## 5. Étape 3 — Re-ranking (Cross-Encoder)

**Fichier** : `core/reranker.py` — classe `CrossEncoderReranker`

### Problème du bi-encoder

FAISS + BM25 utilisent tous les deux un **bi-encoder** : la question et le document sont encodés **séparément**, et comparés par cosine ou BM25. C'est rapide mais moins précis : le modèle n'a jamais vu la question et le document ensemble.

### Solution : Cross-Encoder

Un cross-encoder prend la paire `(question, chunk_text)` en entrée et calcule un score de pertinence en traitant les deux textes **conjointement** via self-attention :

```
Input:  [CLS] Question [SEP] Chunk text [SEP]
            │
            ▼
    Transformer (12 couches)
    (attention croisée question↔chunk)
            │
            ▼
    Score de pertinence (logit unique)
```

### Modèle utilisé

**`BAAI/bge-reranker-v2-m3`** — modèle multilingual de BAAI (mêmes auteurs que BGE-M3). C'est un des meilleurs rerankers publics sur les benchmarks multilingues.

Fallback si FlagEmbedding indisponible : `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`

### Pipeline de reranking

```python
# 1. Extraire les textes des candidats
texts = [result.text for result in results]  # top-20 du hybrid

# 2. Scorer toutes les paires (question, chunk)
pairs = [(question, text) for text in texts]
raw_scores = model.compute_score(pairs)   # liste de logits

# 3. Normalisation sigmoid (logit → probabilité 0-1)
scores = [1 / (1 + exp(-s)) for s in raw_scores]

# 4. Injecter rerank_score dans chaque résultat
for result, score in zip(results, scores):
    result["rerank_score"] = score
    result["score"] = score             # remplace le fused_score
    result["strategy"] += "+rerank"    # "hybrid_rrf+rerank"

# 5. Tri par rerank_score décroissant
results.sort(key=lambda r: r.rerank_score, reverse=True)

# 6. Retourner top-k
return results[:top_k]
```

### Pourquoi le reranker est après le hybrid et pas avant ?

Le cross-encoder est lent (~50-200ms pour 20 paires sur CPU). On ne peut pas l'appliquer à tous les millions de chunks. La stratégie est :

1. **FAISS + BM25** : candidats rapides (pool de 100), rappel maximal
2. **Reranker** : précision maximale sur le petit pool

C'est le paradigme **retrieve → rerank** standard en RAG.

---

## 6. Étape 4 — GraphRAG (optionnel)

**Fichier** : `core/graph_search.py` — méthode `GraphSearch.ask()`

Si `graph_enabled=True`, le résultat du hybrid retrieval est complété par des informations extraites du graphe de connaissances.

### Classifieur d'intention (mode="auto")

```python
def _classify_mode(question: str) -> str:
    q = question.lower()
    # Résumé/vue globale → chercher dans les community summaries
    if any(kw in q for kw in ["main themes", "summarize", "overview", "key topics"]):
        return "global"
    # Question sur une entité spécifique → chercher dans le voisinage local
    if any(kw in q for kw in ["who is", "what did X say", "where is"]):
        return "local"
    # Sinon : hybride graph + vector
    return "hybrid"
```

### Mode "hybrid" (le plus courant)

```
GraphSearch.ask(question, mode="auto")
        │
        ├──► 1. Extraire les entités nommées de la question
        │         ex: ["BB84", "Bennett", "cryptographie quantique"]
        │
        ├──► 2. GraphStore.local_search(seed_entities=["BB84"], hops=1)
        │         → nœuds voisins à 1-2 sauts dans le graphe NetworkX
        │         → retourne chunk_ids associés à ces entités
        │
        ├──► 3. Récupérer les textes des chunks correspondants dans VideoStore
        │
        ├──► 4. GraphStore.global_search(query, top_k=3)
        │         → cosine sur les community summaries
        │         → retourne les 3 résumés de communautés les plus pertinents
        │
        └──► 5. Construire le graph_context pour le LLM :
                  - Liste d'entités et leurs descriptions
                  - Relations entre entités
                  - Résumés de communautés
```

### Ce que le graphe apporte en plus du vector retrieval

| Scénario | Vector seul | Avec graphe |
|----------|-------------|-------------|
| "Quelle est la relation entre BB84 et RSA ?" | Trouve les chunks où les deux sont mentionnés | Trouve directement la relation `BB84 compares_to RSA` dans le graphe |
| "Qui a présenté la théorie X ?" | Cherche "présenté X" sémantiquement | Trouve l'entité PERSON → works_at → ORG + citation |
| "Quels sont les thèmes principaux ?" | Top-k chunks (peut rater les sujets peu denses) | Community summaries LLM → vue structurée |
| "Qu'a dit Alice Chen sur le quantum ?" | Cherche "Alice Chen quantum" | Entité PERSON=Alice Chen → chunks liés → filtre par speaker |

---

## 7. Étape 5 — Vision / VQA (optionnel)

Si `vision=True` dans la requête et que `VQAEngine` est disponible :

```python
# 1. Encoder la question avec CLIP (512d text vector)
query_clip_vec = clip.embed_text(question)

# 2. Rechercher les frames visuelles similaires dans VideoStore
#    (chunk_type="visual", stockées avec leurs vecteurs CLIP)
visual_hits = store.search(query_embedding, chunk_type="visual", top_k=12)

# 3. Pour les top-4 frames les plus proches :
for frame in visual_hits[:4]:
    vlm.vqa(frame.path, question)  → VQAResult(answer, confidence, ...)

# 4. Agréger les réponses visuelles
```

Ce canal est particulièrement utile pour des questions du type :
- "Qu'est-ce qui était écrit sur le tableau à 05:30 ?"
- "Montre-moi la slide avec la formule RRF"
- "Quelle était la courbe montrée à la fin ?"

---

## 8. Étape 6 — Génération de la réponse (LLM)

**Fichier** : `core/llm_client.py` — `synthesize_answer()` ou `stream_answer()`

### Construction du prompt

Le LLM reçoit un prompt structuré en plusieurs blocs :

```
[SYSTEM PROMPT]
You are an expert video analyst who has watched this specific video...
The video is titled "{title}" and covers the domain of "{domain}".
Always answer in {language}. Always include timestamps [MM:SS].

[VIDEO CONTEXT BLOCK] (v6.0 — video-aware)
─────────────────────────────────────────
VIDEO: {title}
SUMMARY: {summary}
DOMAIN: {domain}
KEYWORDS: {keywords}
TOPICS: {topics}
─────────────────────────────────────────

[USER QUESTION]
Question: {question}

[RETRIEVED CHUNKS]
─── Chunk 1 [{start_ts} - {end_ts}] | Speaker: {speaker} | Score: {score:.2f}
{chunk_text}
─── Chunk 2 ...
...

[GRAPH CONTEXT] (si disponible)
Entities mentioned: {entities}
Communities: {community_summaries}

[VISUAL EVIDENCE] (si vision=True)
Frame at {ts}: {vlm_caption}
```

### Détection d'intention et du format de réponse

```python
intent = _detect_intent(question)
# "factual"  → réponse courte et précise avec timestamp
# "summary"  → résumé global, plusieurs paragraphes
# "temporal" → timestamp précis requis, injecter [MM:SS] dans la réponse
# "list"     → format liste à puces
# "yes_no"   → "Oui/Non, car..."
# "translate"→ traduire le contenu dans la langue demandée
```

### Paramètres Ollama

```python
{
    "model": "qwen3.5:2b",
    "stream": False,
    "options": {
        "num_predict": 512,       # v6.0 : 384 → 512 tokens
        "temperature": 0.3,
        "top_p": 0.85,
        "repeat_penalty": 1.15,
        "think": False,           # IMPORTANT pour Qwen3/3.5 — désactive le mode réflexion
    }
}
```

**`think: False`** : les modèles Qwen3 ont un mode "thinking" (chain-of-thought) activable. En production, ce mode augmente massivement la latence et les tokens. Il est désactivé ici.

### Sélection du modèle

```python
PREFERRED_MODEL_PRIORITY = [
    "qwen3.5:2b",   # premier choix — bon rapport qualité/vitesse
    "qwen3:4b",
    "qwen3:8b",
    "qwen2.5:7b",
    "mistral:7b",
    "llama3.1:8b",
    ...
]
# Le premier modèle disponible dans Ollama est utilisé
```

### Extraction des citations

Parallèlement à la génération, le système construit les citations :

```python
citations = []
for result in top_results[:answer_max_evidence]:  # max 4
    citation = _make_citation(result, question)
    # Pour les questions temporelles → _find_precise_span()
    #   cherche dans les word-level timings le mot exact mentionné
    citations.append(citation)
```

**Citation structure** :
```json
{
    "chunk_id": "abc123",
    "video_id": "xyz789",
    "start_time": 324.5,
    "end_time": 341.2,
    "start_ts": "05:24",
    "end_ts": "05:41",
    "speaker": "SPEAKER_00",
    "text": "Le protocole BB84 utilise la polarisation...",
    "score": 0.847,
    "chunk_type": "speech"
}
```

---

## 9. Étape 7 — Citations et post-processing

### `_find_precise_span()` — Timestamps précis au niveau mot

Pour les questions temporelles (intent="temporal"), le système raffine le timestamp en analysant les `word-level timings` produits par WhisperX :

```python
# Exemple: question = "when did he mention RSA ?"
keywords = ["rsa"]

# Dans le chunk [05:20 - 05:45], word timings:
# [{word:"BB84", start:320.1, end:320.5},
#  {word:"compares",start:320.6,...},
#  {word:"RSA",  start:321.2, end:321.4},  ← match !
#  ...]

# Résultat: timestamp précis = 05:21 (au lieu de 05:20)
```

### Post-processing de la réponse

```python
# Si l'intent est "temporal" et qu'un timestamp précis a été trouvé :
answer = _inject_primary_timestamp(answer, "05:21", language_hint)
# → Injecte "[05:21]" dans la réponse générée, remplace le timestamp approximatif
```

---

## 10. Étape 8 — Sauvegarde dans le cache

```python
if _query_cache is not None and request.use_cache:
    _query_cache.save(
        question=question,
        video_id=video_id,
        response=full_response_dict,
        question_embedding=query_emb.bge_dense,
    )
```

La prochaine question similaire (cosine ≥ 0.85) obtiendra cette réponse instantanément.

---

## 11. Mode streaming SSE

`POST /ask/stream` utilise `StreamingResponse` avec le format Server-Sent Events.

### Séquence d'événements SSE

```
event: meta                          ← avant le premier token
data: {
    "video_id": "abc123",
    "title": "Mon cours de crypto",
    "intent": "factual",
    "language": "fr",
    "model": "qwen3.5:2b",
    "citations": [...]               ← disponibles dès le début
}

event: token                         ← un événement par token Ollama
data: {"token": "Le "}

event: token
data: {"token": "protocole "}

event: token
data: {"token": "BB84 "}

...

event: done                          ← fin du stream
data: {
    "elapsed_seconds": 3.21,
    "total_tokens": 127
}
```

**Pourquoi `event:meta` en premier ?** Le frontend peut afficher le titre de la vidéo, les citations, et l'intent *pendant que les tokens arrivent*, sans attendre la réponse complète. C'est une amélioration UX majeure.

**NDJSON Ollama → SSE** :

Ollama stream renvoie du NDJSON ligne par ligne :
```json
{"model":"qwen3.5:2b","response":"Le ","done":false}
{"model":"qwen3.5:2b","response":"protocole ","done":false}
{"model":"qwen3.5:2b","response":"","done":true}
```

Le `stream_answer()` parse chaque ligne et réémet en SSE.

---

## 12. Pourquoi Hybrid et pas seulement Dense ?

Voici les cas concrets où le Hybrid bat le Dense seul dans ce projet :

### Cas 1 — Noms propres et acronymes arabes

```
Question: "ما هو BB84 ؟"   (Qu'est-ce que BB84 ?)

Dense seul: "BB84" encodé en 1024d ← peut être proche de "protocoles quantiques"
            en général. Risque de trouver des chunks qui parlent de quantique
            mais ne mentionnent jamais "BB84".

BM25:       tokenize("BB84") = ["bb84"]
            → cherche exactement "bb84" dans le corpus arabe
            → trouve les chunks qui mentionnent explicitement "BB84"
            → précision parfaite sur ce terme technique

Hybrid RRF: les chunks qui ont à la fois le sens ET le terme exact arrivent premiers
```

### Cas 2 — Paraphrase sémantique

```
Question: "comment fonctionne la distribution de clés quantiques ?"

Dense:  "distribution clés quantiques" ↔ "key distribution quantum cryptography"
        → cosine élevé même si le chunk dit "QKD" ou "quantum key exchange"
        → trouve les chunks qui utilisent des synonymes

BM25:   cherche "distribution" ET "clés" ET "quantiques"
        → rate les chunks qui disent "quantum key distribution" en anglais
        → rate les chunks qui disent "diffusion de clés" (paraphrase)

Hybrid: le dense attrape les paraphrases, le BM25 ancre sur les termes exacts
        → meilleur rappel + précision
```

### Cas 3 — Question courte vs. corpus long

```
Question: "RSA" (une question très courte)

Dense:  le vecteur de "RSA" est très générique — beaucoup de chunks
        sur la crypto en général auront un cosine similaire

BM25:   "RSA" est un terme rare dans le corpus → IDF élevé
        → les chunks qui mentionnent explicitement "RSA" sont fortement récompensés

Hybrid: BM25 joue le rôle d'ancre précise, dense apporte le contexte sémantique
```

### Cas 4 — Requêtes longues et descriptives

```
Question: "explique en détail la différence entre le chiffrement symétrique
           et asymétrique ainsi que leurs avantages respectifs"

Dense:  excellent — vecteur riche qui capture "symétrique", "asymétrique",
        "avantages", "différence" dans l'espace sémantique

BM25:   tokens = ["chiffrement", "symétrique", "asymétrique", "avantages"]
        → les chunks qui couvrent les 4 termes sont récompensés
        → complémentaire du dense pour les requêtes multi-aspects

Hybrid: combine les deux pour maximiser la couverture
```

---

## 13. Paramètres de performance et tuning

### Pool de candidats

```python
candidate_k = max(top_k * 5, 100)
# Si top_k=8 → candidate_k = max(40, 100) = 100 candidats FAISS + 100 BM25
# Après RRF fusion → ~150 candidats uniques
# Après filtrage → variable
# Après reranking → top-8 final
```

Plus `candidate_k` est grand, meilleur est le rappel mais plus lente est la fusion.

### Paramètres de tuning clés

| Paramètre | Valeur actuelle | Impact |
|-----------|----------------|--------|
| `faiss_hnsw_m` | 32 | Plus grand = meilleur rappel, plus de RAM |
| `faiss_hnsw_ef_search` | 128 | Plus grand = plus précis, plus lent |
| `bm25_k1` | 1.5 | Saturation TF (0 = pas de fréquence, ∞ = TF pur) |
| `bm25_b` | 0.75 | Normalisation longueur (0 = pas de pénalité, 1 = full) |
| `rrf_k` | 60 | Régularisation RRF (plus grand = moins d'écart entre rangs) |
| `fusion_ar_dense` | 0.35 | Poids dense pour l'arabe |
| `fusion_ar_sparse` | 0.65 | Poids BM25 pour l'arabe |
| `score_cache_size` | 500 | Nb de requêtes en cache score (LRU, TTL 30min) |

### Latences typiques sur CPU (i7, pas de GPU)

| Étape | Latence typique |
|-------|----------------|
| QueryCache lookup | ~5ms |
| BGE-M3 embed_query | ~80-150ms |
| FAISS HNSW search (1000 chunks) | ~2ms |
| BM25 search (1000 chunks) | ~5ms |
| RRF fusion | ~1ms |
| Cross-encoder rerank (20 paires) | ~200-500ms |
| GraphRAG local search | ~10-30ms |
| LLM synthesis (qwen3.5:2b, 512 tokens) | ~3-8s |
| **Total sans cache** | **~4-10s** |
| **Total avec cache hit** | **~5-10ms** |

---

## 14. Flux complet en pseudo-code

```python
# POST /ask
def ask_endpoint(request: AskRequest):

    # 0. Clarification (optionnel)
    if not request.confirmed and _is_ambiguous(request.question):
        return _run_clarification_sync(request.question)

    question = request.reformulated_question or request.question

    # 1. Cache sémantique
    if request.use_cache and _query_cache:
        hit = _query_cache.lookup(question, request.video_id)
        if hit:
            return hit  # réponse instantanée

    # 2. Embedding de la requête
    query_emb = embedder.embed_query(question)
    # → QueryEmbedding(bge_dense=[...1024...], expanded_text="...")

    # 3. Hybrid Retrieval
    candidate_k = max(top_k * 5, 100)

    # 3a. FAISS dense search
    lang = detect_language(question)
    dense_hits = faiss[lang].search(query_emb.bge_dense, candidate_k)
    dense_hits += faiss["*"].search(query_emb.bge_dense, candidate_k)  # fallback

    # 3b. BM25 sparse search
    sparse_hits = bm25[lang].search(question, candidate_k)
    sparse_hits += bm25["*"].search(question, candidate_k)             # fallback

    # 3c. Filtrage
    dense_hits = [h for h in dense_hits if _passes(h)]
    sparse_hits = [h for h in sparse_hits if _passes(h)]

    # 3d. RRF Fusion
    dense_w, sparse_w = get_fusion_weights(lang)
    # AR: (0.35, 0.65) | EN: (0.65, 0.35) | FR: (0.50, 0.50)
    fused = rrf_fuse(dense_hits, sparse_hits, dense_w, sparse_w, limit=top_k)
    # → [(chunk_id, fused_score, dense_score, sparse_score, strategy), ...]

    # 3e. Convertir en SearchResult
    results = [to_search_result(cid, score, ds, ss, strat) for ... in fused]

    # 4. Cross-Encoder Reranking
    reranked = reranker.rerank(question, results, top_k=top_k)
    # → results triés par rerank_score, strategy="hybrid_rrf+rerank"

    # 5. GraphRAG (si activé)
    graph_context = None
    if graph_search:
        graph_out = graph_search.ask(question, mode="auto", video_id=...)
        graph_context = graph_out["graph"]

    # 6. VQA (si vision=True)
    visual_evidence = None
    if request.vision and vqa_engine:
        visual_evidence = vqa_engine.answer(question, video_id=...)

    # 7. Métadonnées vidéo
    video_meta = metadata_registry.get(video_id)  # titre, résumé, domaine...

    # 8. Génération LLM
    answer = llm_client.synthesize_answer(
        question=question,
        results=reranked,
        language=detect_language(question),
        video_metadata=video_meta,
        graph_context=graph_context,
        visual_evidence=visual_evidence,
    )
    # → LLMAnswer(answer="...", citations=[...], intent="factual", ...)

    # 9. Save in cache
    _query_cache.save(question, video_id, full_response)

    # 10. Return
    return {
        "answer": answer.answer,
        "citations": answer.citations,
        "intent": answer.intent,
        "language": answer.language_hint,
        "model": answer.model,
        "results": [r.to_dict() for r in reranked],
        "elapsed_seconds": elapsed,
    }
```

---

*Document généré le 2026-05-07.*
