# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the Application

```bash
# Start the FastAPI server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Verify dependencies are correctly installed
python test_setup.py
```

The frontend is served statically from `static/index.html` at `/` and `/app`.

## Required External Services

Ollama must be running before starting the server:

```bash
ollama pull qwen3.5:2b        # Main chat model
ollama pull qwen2.5vl:3b      # Vision model
ollama serve                   # Starts on port 11434
```

ML models (BGE-M3, reranker, CLIP, Whisper) are downloaded automatically on first use.

## Tests

```bash
pytest tests/ -v
pytest tests/test_hybrid_search.py   # single test file
```

## Architecture

Three main pipelines all exposed through `api/main.py` (40+ endpoints):

### 1. Ingestion Pipeline — `POST /upload` or `POST /ingest`

Runs sequentially in a background `ThreadPoolExecutor` (max 2 workers):

```
Transcription (WhisperX + Pyannote diarization)
  → Semantic Chunking (MinilM similarity + LLM topic labels)
    → Embedding (BGE-M3 dense 1024d + BM25 sparse + ColBERT tokens)
      → Storage (FAISS HNSW + BM25 per language: AR/EN/FR)
        → Vision Indexing (keyframes → Qwen VLM captions + CLIP 512d + OCR)
          → GraphRAG Extraction (LLM → entities/relations → NetworkX + Leiden communities)
            → Metadata (LLM → title, summary, keywords stored in video_metadata.json)
```

### 2. Search/QA Pipeline — `POST /ask` or `POST /ask/stream`

```
Question
  → Clarification gate (BUG-7: if confirmed=False and ambiguous → ClarificationResponse)
    → Semantic cache (SQLite cosine ≥ 0.92 → return cached)
      → Hybrid retrieval (FAISS top-20 + BM25 top-20 → RRF fusion)
        → Cross-encoder reranking (BAAI/bge-reranker-v2-m3, top-10)
          → Optional graph context (GraphSearch: local/global/hybrid)
          → Optional visual evidence (VQAEngine: CLIP retrieval + VLM answer)
            → LLM synthesis (Ollama streaming, citations with timestamps)
              → Cache save
```

Streaming uses SSE (`/ask/stream`). Language routing uses different FAISS/BM25 index files per language (AR/EN/FR).

### 3. Management — `/videos`, `/video/{id}`, `/graph/*`, `/vision/*`

Video CRUD, reindexing, GraphRAG graph operations, Ollama model listing.

## Key Files

| File | Role |
|------|------|
| `config/settings.py` | All config via Pydantic Settings from `.env` (~178 options) |
| `core/contracts.py` | Canonical data classes: `TranscriptSegment`, `SemanticChunk`, `SearchResult`, `EmbeddedChunk` |
| `core/storage.py` | FAISS HNSW + BM25 hybrid store; language-specific indexes; BM25 lazy rebuild via `_dirty` flag |
| `core/llm_client.py` | Ollama client; streaming, citation formatting, intent detection, answer synthesis |
| `core/search.py` | `AgenticSearch` — orchestrates retrieval → reranking → graph/vision → LLM |
| `core/reranker.py` | `CrossEncoderReranker` — wraps BAAI/bge-reranker-v2-m3 |
| `core/chunking.py` | Semantic chunker — topic boundary detection + LLM topic labeling |
| `core/graph_extractor.py` | GraphRAG — LLM-based entity/relation extraction from chunks |
| `core/vlm_client.py` | Vision-Language Model client (Ollama backend) |
| `core/query_cache.py` | SQLite semantic cache with cosine similarity lookup |

## Configuration

All settings come from `.env` at project root via `config/settings.py`. Key options:

- `PYANNOTE_ENABLED` — speaker diarization; very slow on CPU (32 min for 55s video), disable for development
- `WHISPER_MODEL` / `WHISPER_DEVICE` / `WHISPER_COMPUTE_TYPE` — ASR settings
- `VISION_ENABLED` — disabling skips keyframe/VLM/CLIP indexing
- `GRAPH_ENABLED` — disabling skips GraphRAG extraction
- `OLLAMA_CHAT_MODEL` — primary LLM (default: `qwen3.5:2b`)
- `FUSION_*_DENSE/SPARSE` — per-language RRF score weights

## Persistence Layout

```
data/
├── faiss_store/         # FAISS indexes (ar/en/fr_dense_*.faiss), BM25 pickles, visual_clip/, query_cache.db, video_metadata.json
├── graph_store/         # graph.pkl (NetworkX), graph.json, communities.json
├── keyframes/           # Extracted JPEG frames
├── vision_descriptions/ # VLM captions JSON
├── audio/               # Extracted audio tracks
├── videos/              # Source video files
└── uploads/             # User uploads staging
```

## Design Patterns

- **Language routing**: All FAISS and BM25 indexes are split by detected language (AR/EN/FR/default). `VideoStore.search` selects the correct index at query time.
- **Graceful degradation**: Every pipeline component (vision, graph, reranker, cache) wraps errors and falls back; never blocks the main search path.
- **Lazy model loading**: All ML models are loaded on first use and cached as module-level globals.
- **Clarification gate**: `POST /ask` accepts a `confirmed` boolean. When `False` and the question is detected as ambiguous, the endpoint returns a `ClarificationResponse` (reformulated question) instead of searching — callers must re-submit with `confirmed=True`.
