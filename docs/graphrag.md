# GraphRAG — How the Knowledge Graph Improves Search & Retrieval Quality

This document explains how the graph layer works in this project and, concretely,
**why it makes answers better** than plain vector search alone.

> Scope note: this is written against the architecture documented in `CLAUDE.md`
> (`core/graph_extractor.py`, `core/graph_search.py`, `core/graph_store.py`,
> and the `data/graph_store/` layout). Function/parameter names below are
> descriptive of the pipeline stages; check the source for exact signatures.

---

## 1. The problem GraphRAG solves

Standard RAG retrieves chunks by **semantic similarity**. That works well for
*"find me the passage that talks about X"*, but it fails on two common query
shapes:

| Query type | Example | Why plain vector search struggles |
|------------|---------|-----------------------------------|
| **Multi-hop / relational** | "How is the speaker in clip A related to the company mentioned at the end?" | The answer is spread across chunks that are *not* individually similar to the question. |
| **Global / thematic** | "What are the main themes across this video?" | No single chunk contains the answer; it must be *summarized* over the whole corpus. |

Vector search returns the *k* most similar chunks and stops. It has no notion of
**entities**, **relationships**, or **the overall structure** of the content.

GraphRAG adds that missing structure: it turns the transcript into a graph of
**entities (nodes)** and **relations (edges)**, then uses that graph to retrieve
*connected* context, not just *similar* context.

---

## 2. How the graph is built (ingestion side)

Graph extraction runs as a stage in the ingestion pipeline, after chunking and
embedding:

```
Semantic Chunks
  → LLM entity/relation extraction   (core/graph_extractor.py)
    → NetworkX graph assembly        (core/graph_store.py)
      → Leiden community detection    → communities.json
        → Persist graph.pkl / graph.json
```

### 2.1 Entity & relation extraction
For each semantic chunk, an LLM is prompted to emit structured triples:

```
(entity_a, RELATION, entity_b)   +  entity types  +  short descriptions
```

Example, from a chunk of transcript:

```
("Dr. Sarah Chen", "WORKS_AT", "Acme Robotics")
("Acme Robotics",  "DEVELOPS", "autonomous drones")
("autonomous drones", "USES", "computer vision")
```

Each node and edge keeps a back-reference to the **chunk(s) / timestamps** it came
from, so anything retrieved through the graph can still be cited with a source
segment.

### 2.2 Graph assembly & deduplication
Extracted entities are merged into a single `NetworkX` graph:
- Duplicate/aliased entities are normalized (e.g. "Acme" ≈ "Acme Robotics") so the
  same real-world thing is one node.
- Edges accumulate **weight / frequency** — relations mentioned repeatedly become
  stronger.

### 2.3 Community detection (Leiden)
The **Leiden algorithm** partitions the graph into **communities** — tightly
connected clusters of entities that tend to represent a single topic or subplot.
For each community, the pipeline generates an LLM **community summary**. These
summaries are what power *global* questions.

Persisted artifacts (`data/graph_store/`):

| File | Contents |
|------|----------|
| `graph.pkl` | The NetworkX graph object (nodes, edges, attributes) |
| `graph.json` | Human-readable export of the same graph |
| `communities.json` | Leiden communities + their LLM summaries |

---

## 3. How the graph is used at query time (retrieval side)

At search time, `core/graph_search.py` (`GraphSearch`) offers **three modes**.
This mirrors Microsoft's GraphRAG local/global split:

### 3.1 Local search — *"tell me about this entity and its neighborhood"*
1. Identify the entities in the question (entity linking against graph nodes).
2. Walk the graph outward from those seed nodes (1–2 hops) to pull in **directly
   related entities and relations**.
3. Gather the source chunks attached to those nodes/edges.
4. Feed that connected sub-context to the LLM alongside the vector hits.

Best for: *"Who did X work with?"*, *"What tools were used for Y?"* — questions
anchored on a specific thing.

### 3.2 Global search — *"summarize / what are the themes"*
1. Skip node-walking; instead use the **community summaries** from
   `communities.json`.
2. Map-reduce over the relevant community summaries to build a corpus-wide answer.

Best for: *"What is this video about?"*, *"What are the main arguments?"* — where
the answer lives in the aggregate, not any single passage.

### 3.3 Hybrid — combine graph context with vector retrieval
The graph context is merged with the normal hybrid retrieval
(FAISS + BM25 → RRF → reranker) before LLM synthesis. This is the default the main
QA pipeline uses when `GRAPH_ENABLED=true`.

---

## 4. Where it plugs into the main QA pipeline

From the Search/QA pipeline in `CLAUDE.md`:

```
Question
  → Hybrid retrieval (FAISS top-20 + BM25 top-20 → RRF fusion)
    → Cross-encoder reranking (top-10)
      → Optional graph context (GraphSearch: local / global / hybrid)   ← THIS LAYER
        → LLM synthesis (citations with timestamps)
```

The graph is an **additive context provider**. It runs *alongside* vector
retrieval, and (per the project's graceful-degradation design) if graph extraction
or search fails, the pipeline silently falls back to vector-only results. The
graph can never block the main search path.

---

## 5. Why it improves quality — the concrete wins

### 5.1 Multi-hop reasoning
Vector search retrieves chunks similar to the *question*. If the answer requires
connecting chunk A → chunk B → chunk C, and B isn't textually similar to the
question, plain RAG misses it. The graph makes those hops explicit as edges, so
the connecting context is retrieved even when it isn't semantically similar to the
query.

### 5.2 Global/thematic questions
Community summaries give the LLM a pre-computed, corpus-wide view. Instead of
cramming 20 random chunks into the context window and hoping the theme emerges,
the model reasons over structured summaries. Answers to *"what's the big picture"*
become coherent instead of fragmentary.

### 5.3 Better precision through disambiguation
Because entities are normalized and typed, the graph helps distinguish *"Apple the
company"* from *"apple the fruit"*, and pulls the right neighborhood — reducing
off-topic chunks that pure embedding similarity would surface.

### 5.4 Higher-quality, still-citable context
Every graph node/edge tracks its originating chunk + timestamp, so graph-sourced
context is **still grounded** — the LLM can cite a real segment. You get the
reasoning benefit of the graph without losing traceability.

### 5.5 Denser context per token
A traversed subgraph packs more *relevant, connected* information into the LLM
context window than an equal number of independently-retrieved chunks, which often
overlap or drift off-topic. Better signal-to-noise → better synthesis.

---

## 6. Trade-offs & tuning

| Concern | Notes |
|---------|-------|
| **Ingestion cost** | Extraction is LLM-heavy (one+ call per chunk) and the slowest optional stage. Toggle with `GRAPH_ENABLED`. |
| **Extraction quality** | The graph is only as good as the LLM's triples. A weak model produces noisy/incorrect edges → garbage-in. Use a capable `OLLAMA_CHAT_MODEL`. |
| **Not always needed** | For simple lookup questions, vector search alone is fine; the graph adds most value on relational and global queries. Hybrid mode gets both. |
| **Staleness** | The graph is built at ingest time. Re-ingest / reindex a video to refresh its graph after content changes. |

---

## 7. Mental model (one-liner)

> **Vector search finds text that *looks like* your question.
> The graph finds text that is *connected to* your question's answer —
> and can summarize the whole video at once.**

Using both, then reranking and synthesizing with citations, is what makes
retrieval quality noticeably higher than plain RAG on relational and thematic
questions.
