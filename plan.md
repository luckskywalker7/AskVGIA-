# Deployment Plan — Free MVP (QA-only, cloud, API LLM)

> Goal: host the frontend + backend fully in the cloud (nothing on the local
> machine), on **free** tiers, where **only the QA function works** via a hosted
> LLM API. Ingestion (Whisper / VLM / pyannote) does **not** need to run — we
> serve pre-built artifacts read-only.

---

## 1. Feasibility verdict

**QA on already-ingested videos, fully cloud, CPU-only, using an LLM API — feasible.** ✅

Why it works:
- `VideoTranscriber` uses `load_on_init=False` (`core/transcription.py:277`) → Whisper
  never loads at startup, and never at all for QA.
- A QA-only server loads only: **BGE-M3 (~2 GB)** + reranker (~1 GB) + FAISS/BM25 in
  RAM + torch overhead ≈ **5–7 GB resident**. Fits in a 16 GB free box.

**Hard boundary — do NOT try to ingest in the cloud without a GPU.**
Ingestion (Whisper + VLM captioning + pyannote) needs a GPU (settings note ~27 s/frame
even *on* a 16 GB GPU). On a CPU box it is effectively impossible. We avoid this entirely
by shipping the **prebuilt `data/`** and only answering questions.

- `data/` is already ingested (present in the repo working tree).
- If a new video ever needs ingesting: rent a GPU pod (RunPod / Vast, ~$0.30/hr) once,
  ingest, copy `data/` out, tear it down. Never on the free CPU host.

---

## 2. The key reframing: read-only ⇒ no paid storage

Because QA is **read-only**, we do **not** need a paid persistent volume. We **bake
`data/` into the container image / Space repo**. Any stateless free host works — the
only requirement is enough **RAM**.

RAM is the whole game: QA must embed the query with the **same BGE-M3 model** that built
the FAISS vectors, so BGE-M3 (~2 GB) is non-negotiable. That kills almost every free tier.

| Free host | RAM | Verdict |
|---|---|---|
| Render free | 512 MB | ❌ Can't load BGE-M3 |
| Fly.io / Koyeb free | 256–512 MB | ❌ Too small |
| Railway | $5 trial only | ❌ Not really free |
| Google Cloud Run free | ~512 MB, scale-to-zero | ❌ Cold-loads 2 GB model per request |
| **Hugging Face Spaces (Docker, CPU basic)** | **16 GB free** | ✅ **Recommended** |
| **Oracle Cloud Always Free (ARM Ampere)** | **24 GB free forever** | ✅ Real VM, more control |

---

## 3. Recommended stack (simplest free MVP)

**One deploy: Hugging Face Space (Docker SDK, free CPU basic).**

- 16 GB RAM fits BGE-M3 + reranker + FAISS/BM25 comfortably.
- Bake `data/faiss_store/` + `data/graph_store/` into the Space repo (git-LFS) or the image
  → no persistent volume, no paid storage.
- Public URL out of the box.
- FastAPI already mounts the static frontend at `/` and `/app` (`api/main.py:145-150`), so
  **the Space serves BOTH frontend and backend** → no CORS, no relative-path problem, one
  thing to deploy.
- Free Spaces sleep after ~48 h idle, cold-start ~1–2 min. Fine for an MVP demo.

**LLM for QA:** **Groq free API** (recommended: free, OpenAI-compatible, very fast) or
**Google Gemini free tier**. Requires the one code edit in §6.

**Videos:** for the MVP, **drop the video player** (QA answers + timestamp citations are the
product; videos are the big files that bloat the image). Add later via **Cloudflare R2
(free 10 GB)** if wanted. Not needed for QA to work.

### Alternative: split frontend / backend
- Frontend: Vercel / Cloudflare Pages / GitHub Pages (free static) — deploy `static/app/`,
  add `vercel.json` rewrites proxying to the backend URL (see §7).
- Backend: same HF Space, or an **Oracle Always-Free 24 GB ARM VM** for an always-on server
  (no sleep) with room to keep videos too.

For a *simple MVP*, the all-in-one HF Space is the least work.

---

## 4. Difficulties to expect

| # | Difficulty | Mitigation |
|---|---|---|
| 1 | **LLM API swap is not env-only** | `core/llm_client.py` speaks Ollama's `/api/chat` + `/api/generate` and reads `message.content` / Ollama NDJSON. OpenAI/Groq/Gemini use a different shape → must edit 3 functions (§6). |
| 2 | **SSE streaming buffered/killed** | `/ask/stream` is Server-Sent Events. Serverless & some proxies buffer or time out. → Use a persistent host (HF Space / VM), not scale-to-zero serverless. |
| 3 | **Persistent disk normally required** | Sidestepped by baking read-only `data/` into the image. If you ever write (cache/history), that write is lost on restart — acceptable for MVP. |
| 4 | **Model downloads on cold start** | First boot pulls BGE-M3 + reranker (~3 GB) from HuggingFace. → Set `HF_HOME` to a cached path; on HF Spaces this is fast (same network). |
| 5 | **QA latency on CPU** | Reranker ≈ 3 s/chunk on CPU + embedding + LLM round-trip → expect 10–30 s/answer. Lower `answer_max_context_chunks` / rerank top-k if needed. |
| 6 | **RAM is the cost driver** | 5–7 GB resident before concurrency. Few concurrent users on one box (state on local disk, no easy horizontal scale). Size for 16 GB. |
| 7 | **Vision VQA wants a VLM** | Frontend sends `vision:true` by default (`static/app/js/api.js:90,108`). Disable vision (`VISION_ENABLED=false`) → QA answers from speech; no VLM needed. |
| 8 | **Video bandwidth** | Videos stream through the backend (`/video/{id}/stream`, Range). Drop for MVP or offload to R2. |
| 9 | **CORS + cross-origin auth** | Backend sets `allow_origins=["*"]` with `allow_credentials=True` (`api/main.py:119`) — invalid combo for cookies. All-in-one Space (same-origin) avoids this. |
| 10 | **`WHISPER_DEVICE` default is `cuda`** | `config/settings.py:24`. Harmless for QA (never loads); set `WHISPER_DEVICE=cpu` to be safe. |

---

## 5. Where each piece lives

- **Frontend (cloud):** served by the Space at `/` and `/app`, OR Vercel/Cloudflare Pages
  (static `static/app/`).
- **Videostore (FAISS + BM25 + metadata):** baked into image at `data/faiss_store/`
  (`storage_dir`, `config/settings.py:44`).
- **Graph (graph.pkl / graph.json / communities.json):** baked in at `data/graph_store/`
  (`graph_store_dir`, `config/settings.py:252`).
- **Videos:** skip for MVP, or Cloudflare R2 (free 10 GB). Served today from `data/videos/`
  via `/video/{id}/stream` (`api/main.py:2255`).

---

## 6. The one code change — Ollama → API LLM (QA only)

All QA text generation is in `core/llm_client.py`, speaking Ollama's native API. To use a
hosted API (Groq/Gemini/OpenAI), edit these three call sites (nothing else):

| Function | Location | Role |
|---|---|---|
| `_call_ollama_chat()` | `core/llm_client.py:2155` (POST `/api/chat`) | Non-streaming answer synthesis |
| stream block in `stream_answer()` | `core/llm_client.py:2429-2448` (POST `/api/chat`, `stream=True`) | `/ask/stream` SSE answers |
| `chat_plain()` | `core/llm_client.py:3025` (POST `/api/chat`) | Modality classifier / clarification |

Supporting helpers: `_get_ollama_base_url()` (`:306`), `_choose_model()` / model list via
GET `/api/tags` (`:~2094`). `chat_json()` (`:3077`, `/api/generate`) is used by JSON helpers;
edit too if those paths are exercised.

Config knobs (via `.env`): `OLLAMA_BASE_URL` (`config/settings.py:219`),
`OLLAMA_CHAT_MODEL` (`config/settings.py:238`).

**Recommended provider: Groq** — free, OpenAI-compatible, fast. The edit swaps the Ollama
request/response shape for `/v1/chat/completions` (read `choices[0].message.content`, and for
streaming read SSE `delta.content`). ~30 lines total, isolated to `llm_client.py`.

> STATUS: not yet done. Decide Groq vs Gemini, then implement.

---

## 7. Optional: `vercel.json` (only if splitting frontend to Vercel)

```json
{
  "rewrites": [
    { "source": "/videos/:path*",  "destination": "https://BACKEND_URL/videos/:path*" },
    { "source": "/video/:path*",   "destination": "https://BACKEND_URL/video/:path*" },
    { "source": "/ask/:path*",     "destination": "https://BACKEND_URL/ask/:path*" },
    { "source": "/ask",            "destination": "https://BACKEND_URL/ask" },
    { "source": "/search",         "destination": "https://BACKEND_URL/search" },
    { "source": "/graph/:path*",   "destination": "https://BACKEND_URL/graph/:path*" },
    { "source": "/vision/:path*",  "destination": "https://BACKEND_URL/vision/:path*" },
    { "source": "/history/:path*", "destination": "https://BACKEND_URL/history/:path*" },
    { "source": "/health",         "destination": "https://BACKEND_URL/health" }
  ]
}
```

Not needed for the all-in-one HF Space (frontend served same-origin).

---

## 8. `.env` for QA-only (strip everything except QA)

```env
# LLM (QA) — set after the llm_client edit
OLLAMA_BASE_URL=https://api.groq.com/openai        # or provider base
OLLAMA_CHAT_MODEL=llama-3.3-70b-versatile           # or chosen free model
# GROQ_API_KEY / GEMINI_API_KEY as required by the edit

# Turn OFF everything not needed for QA
VISION_ENABLED=false          # no CLIP, no VLM
PYANNOTE_ENABLED=false
WHISPER_DEVICE=cpu            # never loads for QA; guarantees no CUDA attempt

# Keep prebuilt read-only artifacts
GRAPH_ENABLED=true

# Model cache on the container FS
HF_HOME=/app/.hf_cache
```

---

## 9. Step-by-step (all-in-one HF Space)

1. Create a Hugging Face **Space** → SDK: **Docker**, hardware: **CPU basic (free)**.
2. Add a `Dockerfile` that installs `requirements.txt` (CPU torch — drop the `+cu128`
   suffix / `--extra-index-url` cuda line), copies the repo, and runs
   `uvicorn api.main:app --host 0.0.0.0 --port 7860`.
3. Commit `data/faiss_store/` + `data/graph_store/` via **git-LFS** (exclude `data/videos/`
   for MVP).
4. Do the `core/llm_client.py` edit (§6) for Groq/Gemini; add the API key as a Space
   **secret**.
5. Add the `.env` values (§8) as Space **variables/secrets**.
6. Push. Space builds, boots, downloads BGE-M3 + reranker once, health check passes.
7. Open the Space URL → pick an already-ingested video → ask questions. Streaming answers,
   citations, and graph all work; ingestion endpoints exist but are unused.

---

## 10. Open decisions

- [ ] **LLM provider:** Groq (recommended) or Gemini?
- [ ] Keep video playback (needs R2 / bandwidth) or drop for MVP?
- [ ] All-in-one HF Space, or split Vercel frontend + Space backend?

### Remaining work
1. Edit `core/llm_client.py` (3 call sites) for the chosen API. ← only real code task
2. Write `Dockerfile` + CPU `requirements` tweak.
3. Set up the HF Space, LFS-commit `data/`, add secrets, deploy.
