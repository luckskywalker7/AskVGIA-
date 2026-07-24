"""
qa_service.py — AskVGIA cloud QA API.

Stateless. No local models, no FAISS, no Ollama. Everything it needs lives in
Supabase or behind an HTTP API, so it runs on a small free host with your PC off.

  question -> HuggingFace BGE-M3 (1024-d)   [same model your vectors came from]
           -> Supabase hybrid_search()      [pgvector cosine + Postgres FTS + RRF]
           -> OpenRouter free LLM           [answer + citations]

Run locally:
    pip install fastapi uvicorn httpx supabase python-dotenv
    uvicorn qa_service:app --reload --port 8100

Env vars required:
    HF_TOKEN               your existing HuggingFace token
    OPENROUTER_API_KEY     openrouter.ai key
    SUPABASE_URL           https://<ref>.supabase.co
    SUPABASE_ANON_KEY      anon public key (read-only tables, RLS allows select)
"""

import os
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load .env from the project root so the env vars below resolve when you run
# `uvicorn qa_service:app` without exporting them by hand. No-op if the package
# isn't installed or there is no .env.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------- config
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://zievhmocgbrrsltnyfkf.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

HF_EMBED_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-m3/pipeline/feature-extraction"

# Free model IDs rotate constantly on OpenRouter (models get retired without
# notice). "openrouter/free" is their auto-router: it picks whatever free model
# is currently available. That's what keeps this working unattended for months.
# The named IDs after it are only a backstop; verify at openrouter.ai/models
# if you ever want to pin a specific one.
OPENROUTER_MODELS = [
    "openrouter/free",
    "openai/gpt-oss-120b:free",
    "openai/gpt-oss-20b:free",
]

TOP_K = 10          # chunks fed to the LLM
CANDIDATES = 60     # per-branch candidates inside hybrid_search

app = FastAPI(title="AskVGIA QA")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ----------------------------------------------------------------- models
class AskRequest(BaseModel):
    video_id: str
    question: str
    language: Optional[str] = None      # 'auto' | 'en' | 'fr' | 'ar'


class Citation(BaseModel):
    chunk_id: str
    text: str
    start_time: float
    end_time: float
    chunk_type: str
    speaker: Optional[str] = None
    frame_url: Optional[str] = None
    youtube_url: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    video_id: str
    youtube_id: Optional[str] = None
    model_used: Optional[str] = None


# ----------------------------------------------------------------- helpers
def _sb_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }


async def embed_question(text: str) -> List[float]:
    """BGE-M3 dense 1024-d via HF Inference API, with cold-start retries."""
    if not HF_TOKEN:
        raise HTTPException(500, "HF_TOKEN not set")

    payload = {"inputs": text, "options": {"wait_for_model": True}}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    last = None
    async with httpx.AsyncClient(timeout=90) as client:
        for attempt in range(4):
            try:
                r = await client.post(HF_EMBED_URL, json=payload, headers=headers)
                if r.status_code == 503:      # model cold-loading
                    await asyncio.sleep(4 * (attempt + 1))
                    continue
                r.raise_for_status()
                vec = r.json()
                # HF may return [1024] or [[1024]] depending on the pipeline
                while isinstance(vec, list) and vec and isinstance(vec[0], list):
                    vec = vec[0]
                if not isinstance(vec, list) or len(vec) != 1024:
                    raise ValueError(f"unexpected embedding shape: {len(vec) if isinstance(vec,list) else type(vec)}")
                return [float(x) for x in vec]
            except Exception as e:
                last = e
                await asyncio.sleep(2 * (attempt + 1))
    raise HTTPException(502, f"embedding failed: {last}")


async def hybrid_search(video_id: str, question: str, vec: List[float]) -> List[Dict[str, Any]]:
    """Call the Supabase RPC — dense + FTS + RRF, filtered to one video."""
    url = f"{SUPABASE_URL}/rest/v1/rpc/hybrid_search"
    body = {
        "p_video_id": video_id,
        "p_query_text": question,
        "p_query_vec": vec,
        "p_limit": TOP_K,
        "p_candidates": CANDIDATES,
    }
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(url, json=body, headers=_sb_headers())
        if r.status_code >= 400:
            raise HTTPException(502, f"retrieval failed: {r.text[:200]}")
        return r.json()


async def get_video(video_id: str) -> Dict[str, Any]:
    url = f"{SUPABASE_URL}/rest/v1/demo_videos"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_sb_headers(),
                             params={"video_id": f"eq.{video_id}", "select": "*"})
        rows = r.json() if r.status_code < 400 else []
    return rows[0] if rows else {}


def fmt_ts(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 60}:{sec % 60:02d}"


def build_context(rows: List[Dict[str, Any]]) -> str:
    """Numbered passages the model can cite by index."""
    parts = []
    for i, r in enumerate(rows, 1):
        md = r.get("metadata") or {}
        body = r.get("text") or ""
        # for visual chunks the on-screen text is the most citable evidence
        if r.get("chunk_type") == "visual":
            extra = " ".join(filter(None, [
                md.get("board_text"), md.get("screen_text"), md.get("ocr_text"),
            ]))[:400]
            cap = md.get("caption") or md.get("short_caption") or ""
            body = " | ".join(filter(None, [extra, cap, body]))[:600]
        parts.append(
            f"[{i}] ({fmt_ts(r.get('start_time'))}–{fmt_ts(r.get('end_time'))}, "
            f"{r.get('chunk_type')}) {body}"
        )
    return "\n\n".join(parts)


SYSTEM = """You answer questions about a single video using only the numbered passages provided.

Rules:
- Ground every claim in the passages. If they don't contain the answer, say so plainly.
- Cite passages inline as [1], [2] — cite the specific passage each claim comes from.
- Answer in the same language as the question.
- Passages marked "visual" come from on-screen text and frame descriptions; passages marked "speech" come from the spoken transcript. Both are valid evidence.
- Be concise and direct. No preamble, no restating the question."""


async def generate_answer(question: str, context: str) -> tuple[str, str]:
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY not set")

    user = (
        f"Question: {question}\n\n"
        f"Passages from the video:\n\n{context}\n\n"
        f"Answer the question using only these passages, citing them as [n]."
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://askvgia.app",
        "X-Title": "AskVGIA",
    }

    last = None
    async with httpx.AsyncClient(timeout=120) as client:
        for model in OPENROUTER_MODELS:
            try:
                r = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 700,
                    },
                )
                if r.status_code >= 400:
                    last = f"{model} -> {r.status_code}: {r.text[:180]}"
                    continue
                data = r.json()
                if "error" in data:
                    last = f"{model} -> {str(data['error'])[:180]}"
                    continue
                txt = data["choices"][0]["message"]["content"].strip()
                if txt:
                    return txt, data.get("model", model)
            except Exception as e:
                last = f"{model} -> {type(e).__name__}: {str(e)[:180]}"
                continue
    raise HTTPException(502, f"no free model answered. last error: {last}")


# ----------------------------------------------------------------- routes
@app.get("/health")
async def health():
    return {
        "ok": True,
        "supabase": bool(SUPABASE_ANON_KEY),
        "hf": bool(HF_TOKEN),
        "openrouter": bool(OPENROUTER_API_KEY),
    }


@app.get("/videos")
async def videos():
    url = f"{SUPABASE_URL}/rest/v1/demo_videos"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_sb_headers(),
                             params={"select": "*", "order": "n_chunks.desc"})
        if r.status_code >= 400:
            raise HTTPException(502, r.text[:200])
        return r.json()


@app.get("/graph/{video_id}")
async def graph(video_id: str):
    url = f"{SUPABASE_URL}/rest/v1/rpc/get_video_graph"
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(url, json={"p_video_id": video_id}, headers=_sb_headers())
        if r.status_code >= 400:
            raise HTTPException(502, r.text[:200])
        return r.json()


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "question is empty")

    video = await get_video(req.video_id)
    youtube_id = video.get("youtube_id")

    vec = await embed_question(q)
    rows = await hybrid_search(req.video_id, q, vec)

    if not rows:
        return AskResponse(
            answer="I couldn't find anything in this video that answers that.",
            citations=[], video_id=req.video_id, youtube_id=youtube_id,
        )

    context = build_context(rows)
    answer, model_used = await generate_answer(q, context)

    cites = []
    for r in rows:
        md = r.get("metadata") or {}
        st = float(r.get("start_time") or 0)
        cites.append(Citation(
            chunk_id=r.get("chunk_id", ""),
            text=(r.get("text") or "")[:400],
            start_time=st,
            end_time=float(r.get("end_time") or 0),
            chunk_type=r.get("chunk_type") or "speech",
            speaker=r.get("speaker"),
            frame_url=md.get("frame_url"),
            youtube_url=(f"https://www.youtube.com/watch?v={youtube_id}&t={int(st)}s"
                         if youtube_id else None),
        ))

    return AskResponse(
        answer=answer, citations=cites, video_id=req.video_id,
        youtube_id=youtube_id, model_used=model_used,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
