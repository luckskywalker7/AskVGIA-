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
import json
import base64
import asyncio
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

# Load .env from the project root so the env vars below resolve when you run
# `uvicorn qa_service:app` without exporting them by hand. No-op if the package
# isn't installed or there is no .env. (On Render the platform injects them, so
# this is only needed for local runs.)
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
#
# The named IDs after it are only a backstop, and they MUST be checked against
# https://openrouter.ai/api/v1/models before being listed here. The two previous
# entries (openai/gpt-oss-120b:free, openai/gpt-oss-20b:free) had been retired
# and returned 404 "This model is unavailable for free" on every single call.
# Because the loop below kept only the LAST error, that dead backstop became the
# error users saw ("model not available") whenever the router itself hiccuped,
# masking the real cause. Both IDs below were confirmed live (HTTP 200, correct
# French reply) and are listed at zero price in the catalogue.
# openrouter/free appears three times on purpose. It picks a free model at
# RANDOM on every single call, so a second attempt is a genuinely different
# model — by far the cheapest way to survive one bad draw (a model that returns
# empty content, or an upstream 429). The two pinned IDs are the last resort.
OPENROUTER_MODELS = [
    "openrouter/free",
    "openrouter/free",
    "openrouter/free",
    "nvidia/nemotron-3.5-lightning:free",
    "liquid/lfm-2.5-2.6b:free",
]

# Most free models are reasoning models: they fill a "reasoning" field before
# writing "content". Providers differ on whether those tokens count against this
# budget, and when they do, 700 ran out mid-thought and content came back null
# with finish_reason "length". Free models cost nothing per token, so the only
# thing a generous ceiling buys is headroom — the answers themselves stay short
# because the prompt asks for 2-5 points.
MAX_ANSWER_TOKENS = 6000

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
    language: Optional[str] = None
    # accepted and ignored — the existing frontend sends these
    confirmed: Optional[bool] = None
    use_llm: Optional[bool] = None
    use_rerank: Optional[bool] = None
    top_k: Optional[int] = None
    vision: Optional[bool] = None


class Citation(BaseModel):
    chunk_id: str
    text: str
    start_time: float
    end_time: float
    chunk_type: str
    speaker: Optional[str] = None
    frame_url: Optional[str] = None
    youtube_url: Optional[str] = None
    # aliases used by the existing frontend renderers
    start: float = 0
    end: float = 0
    score: float = 0
    video_id: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AskResponse(BaseModel):
    answer: str
    citations: List[Citation]
    # aliases the existing frontend reads (api.js: data.chunks || data.sources)
    chunks: List[Citation] = []
    sources: List[Citation] = []
    results: List[Citation] = []
    video_id: str
    youtube_id: Optional[str] = None
    model_used: Optional[str] = None
    # which language the answer was written in, and whether it was asked for
    # explicitly ("requested") or inferred from the question ("detected")
    language: Optional[str] = None
    language_source: Optional[str] = None
    # "hybrid" = dense + FTS as designed; "keyword" = embeddings were down and
    # this answer came from full-text search only. The UI can warn on the latter.
    retrieval_mode: Optional[str] = None
    notice: Optional[str] = None


# ----------------------------------------------------------------- helpers
def _sb_headers() -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
    }


class EmbeddingUnavailable(Exception):
    """
    Dense embeddings could not be produced. Carries whether the cause is
    transient (cold start, rate limit) or terminal (no credits, bad token) so
    /ask can say something true instead of always blaming a warmup.
    """

    def __init__(self, detail: str, transient: bool = False):
        super().__init__(detail)
        self.detail = detail
        self.transient = transient


# ----------------------------------------------------------------- language
# core/language_detector.py does this properly, but importing it would drag
# core.contracts (and optionally langdetect) into the cloud image, and
# requirements-cloud.txt is deliberately four packages. Same heuristics,
# inlined: Arabic script by codepoint first, then FR/EN stopword counts.
def detect_language(text: str) -> str:
    """
    Return "ar", "fr" or "en" for a question. Mirrors LanguageDetector's
    ordering: Arabic script wins on any meaningful presence (it is never
    produced by accident), then stopword counts, then English.
    """
    import re
    t = (text or "").strip()
    if not t:
        return "en"

    arabic = len(re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", t))
    latin = len(re.findall(r"[A-Za-z\u00C0-\u00FF]", t))
    if arabic and arabic >= 0.20 * max(1, arabic + latin):
        return "ar"

    fr = len(re.findall(
        r"\b(le|la|les|des|une|un|est|sont|avec|dans|pour|sur|pas|plus|que|qui|"
        r"quoi|quel|quelle|comment|pourquoi|quand|ce|cette|du|au|aux|et|ou|mais|"
        r"vidone|parle|dit|explique|resume)\b", t, re.I))
    en = len(re.findall(
        r"\b(the|a|an|is|are|was|were|with|in|for|on|not|more|than|what|how|why|"
        r"when|which|this|that|of|and|or|but|does|do|did|say|says|explain|"
        r"summarize|summarise|about)\b", t, re.I))
    if fr > en:
        return "fr"
    if en > fr:
        return "en"
    # Accents are a French tell when the stopword counts tie.
    if re.search(r"[\u00e0\u00e2\u00e7\u00e9\u00e8\u00ea\u00eb\u00ee\u00ef\u00f4\u00fb\u00f9\u00fc]", t, re.I):
        return "fr"
    return "en"


LANG_NAMES = {"ar": "Arabic", "fr": "French", "en": "English"}


def language_directive(lang: str) -> str:
    """The one hard instruction that pins the answer language, named outright."""
    name = LANG_NAMES.get(lang, "English")
    line = (
        "ANSWER LANGUAGE: {n}. Write the ENTIRE answer in {n}, including the "
        "opening sentence and every numbered point. Do not answer in any other "
        "language, and do not translate technical terms or on-screen wording out "
        "of the form the video uses.".format(n=name)
    )
    if lang == "ar":
        line += (" Use Arabic script, not transliteration. Leave the timestamp"
                 " digits as Western numerals (2:14) — the interface parses them"
                 " to seek the video.")
    return line


async def embed_question(text: str) -> List[float]:
    """BGE-M3 dense 1024-d via HF Inference API, with cold-start retries."""
    if not HF_TOKEN:
        # Same treatment as a dead endpoint: /ask degrades to keyword retrieval
        # rather than 500-ing, so a missing token costs precision, not service.
        raise EmbeddingUnavailable("HF_TOKEN not set")

    payload = {"inputs": text, "options": {"wait_for_model": True}}
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    last = None
    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(8):
            try:
                r = await client.post(HF_EMBED_URL, json=payload, headers=headers)
                if r.status_code == 503:                 # model cold-loading
                    last = "model loading (503)"
                    await asyncio.sleep(min(8 + attempt * 6, 30))
                    continue
                if r.status_code == 429:                 # rate limited
                    last = "rate limited (429)"
                    await asyncio.sleep(min(6 + attempt * 4, 25))
                    continue
                if r.status_code >= 400:
                    last = f"HF {r.status_code}: {r.text[:160]}"
                    # 402 = the HF account has no Inference credits left. That is
                    # permanent until someone tops up, so retrying is pure
                    # latency: eight attempts burned ~35s and then reported "the
                    # model is warming up", sending users off to wait for a cold
                    # start that was never going to finish. Fail fast instead and
                    # let /ask fall back to keyword retrieval.
                    if r.status_code in (401, 402, 403, 404):
                        raise EmbeddingUnavailable(last)
                    await asyncio.sleep(4)
                    continue
                vec = r.json()
                # HF may return [1024] or [[1024]] depending on the pipeline
                while isinstance(vec, list) and vec and isinstance(vec[0], list):
                    vec = vec[0]
                if not isinstance(vec, list) or len(vec) != 1024:
                    raise ValueError(f"unexpected embedding shape: {len(vec) if isinstance(vec,list) else type(vec)}")
                return [float(x) for x in vec]
            except (HTTPException, EmbeddingUnavailable):
                raise
            except Exception as e:
                last = str(e)[:160]
                await asyncio.sleep(3 + attempt * 2)
    # Genuinely transient: every attempt above hit 503/429 or a network error.
    raise EmbeddingUnavailable(last or "exhausted retries", transient=True)


async def hybrid_search(video_id: str, question: str, vec: List[float],
                        limit: int = TOP_K) -> List[Dict[str, Any]]:
    """Call the Supabase RPC — dense + FTS + RRF, filtered to one video."""
    url = f"{SUPABASE_URL}/rest/v1/rpc/hybrid_search"
    body = {
        "p_video_id": video_id,
        "p_query_text": question,
        "p_query_vec": vec,
        "p_limit": limit,
        "p_candidates": CANDIDATES,
    }
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(url, json=body, headers=_sb_headers())
        if r.status_code >= 400:
            raise HTTPException(502, f"retrieval failed: {r.text[:200]}")
        return r.json()


async def get_video(video_id: str) -> Dict[str, Any]:
    # demo_videos_ext = demo_videos + graph_edges / graph_nodes / has_graph
    url = f"{SUPABASE_URL}/rest/v1/demo_videos_ext"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_sb_headers(),
                             params={"video_id": f"eq.{video_id}", "select": "*"})
        rows = r.json() if r.status_code < 400 else []
    return rows[0] if rows else {}


def fmt_ts(sec: float) -> str:
    sec = int(sec or 0)
    return f"{sec // 60}:{sec % 60:02d}"


def _norm(s: str) -> set:
    """Word set for cheap similarity, ignoring OCR clock/noise digits."""
    import re
    s = re.sub(r"\d{1,2}:\d{2}\s*(am|pm)?", " ", (s or "").lower())
    return set(re.findall(r"[a-z؀-ۿÀ-ſ]{3,}", s))


def dedupe_visual(rows: List[Dict[str, Any]], thresh: float = 0.85) -> List[Dict[str, Any]]:
    """
    Frames are sampled every ~2s, so a slide held on screen for 20s becomes ~10
    near-identical visual chunks with the same OCR text. Keeping all of them
    wastes the context window and makes the citation list look broken.

    Collapse visual chunks whose text is >thresh similar to one already kept,
    keeping the EARLIEST occurrence (when the slide first appeared — also the
    better timestamp to jump the player to). Speech chunks are never deduped.
    """
    kept: List[Dict[str, Any]] = []
    seen: List[set] = []
    for r in sorted(rows, key=lambda x: (x.get("chunk_type") != "speech",
                                         float(x.get("start_time") or 0))):
        if (r.get("chunk_type") or "") != "visual":
            kept.append(r)
            continue
        md = r.get("metadata") or {}
        blob = " ".join(filter(None, [
            r.get("text"), md.get("board_text"), md.get("screen_text"), md.get("ocr_text"),
        ]))
        w = _norm(blob)
        if not w:
            kept.append(r)
            continue
        dup = False
        for prev in seen:
            inter = len(w & prev)
            union = len(w | prev) or 1
            if inter / union >= thresh:
                dup = True
                break
        if not dup:
            seen.append(w)
            kept.append(r)
    # restore relevance order from the original ranking
    order = {r.get("chunk_id"): i for i, r in enumerate(rows)}
    return sorted(kept, key=lambda r: order.get(r.get("chunk_id"), 999))


# Candidates pulled from Postgres before local re-ranking. PostgREST cannot
# ORDER BY ts_rank without a dedicated RPC, so ranking happens here instead;
# 200 rows is a few hundred KB and covers any single video in this corpus.
_KEYWORD_CANDIDATES = 200


def _query_terms(question: str) -> List[str]:
    """
    Split a question into tsquery-safe terms.

    to_tsquery() is strict: ':', '&', '|', '!', '(', ')' and stray quotes make it
    raise, which would turn a fallback into a 500. Keeping only letters, digits
    and Arabic codepoints sidesteps the whole escaping problem.
    """
    import re
    words = re.findall(
        r"[0-9A-Za-z\u00C0-\u00FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+",
        (question or "").lower())
    seen, terms = set(), []
    for w in words:
        if len(w) < 2 or w in seen:
            continue
        seen.add(w)
        terms.append(w)
    return terms[:24]          # a long question should not build a huge tsquery


async def keyword_search(video_id: str, question: str,
                         limit: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Full-text-only retrieval, used when embeddings are unavailable.

    hybrid_search needs a 1024-d query vector, so it cannot run at all without
    the embedding endpoint. This hits PostgREST directly against the indexed
    `fts` tsvector column — no RPC, no vector — so /ask can still answer (less
    precisely) while the embedding provider is down or unfunded.

    Terms are OR-ed, not AND-ed. The obvious `fts=wfts.<question>` is wrong here:
    websearch_to_tsquery ANDs every lexeme, so a whole sentence only matches a
    chunk containing all of it at once — in practice zero rows for any real
    question. OR-ing recovers the matches, then the rows are re-ranked locally by
    how many distinct query terms each one actually contains, because PostgREST
    cannot ORDER BY ts_rank. Without that re-rank the `limit` would just take
    whichever rows Postgres happened to return first.

    Returns rows shaped like the RPC's, with rrf_score carrying the overlap score
    so existing callers stay uniform.
    """
    terms = _query_terms(question)
    if not terms:
        return []

    url = f"{SUPABASE_URL}/rest/v1/video_chunks"
    params = {
        "select": ("chunk_id,video_id,text,start_time,end_time,"
                   "chunk_type,speaker,metadata"),
        "video_id": f"eq.{video_id}",
        "fts": "fts." + "|".join(terms),
        "limit": str(_KEYWORD_CANDIDATES),
    }
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(url, params=params, headers=_sb_headers())
        if r.status_code >= 400:
            # A tsquery Postgres still refuses is not worth failing /ask over.
            print(f"[keyword_search] {r.status_code}: {r.text[:160]}")
            return []
        rows = r.json() or []

    # Rank by distinct-term overlap, longer terms first (they are the content
    # words; "the" and "le" survive tokenising but carry almost no signal).
    import re as _re2
    weights = {t: (2.0 if len(t) > 4 else 1.0) for t in terms}
    for row in rows:
        text = (row.get("text") or "").lower()
        score = sum(w for t, w in weights.items()
                    if _re2.search(r"(?<![0-9a-z])" + _re2.escape(t), text))
        row["rrf_score"] = score
    rows = [r_ for r_ in rows if r_["rrf_score"] > 0]
    rows.sort(key=lambda r_: (-r_["rrf_score"], float(r_.get("start_time") or 0)))
    return rows[:limit]


# No-match text, in the answer language — an English-only string here would
# defeat the whole point of pinning the answer language.
NO_MATCH = {
    "en": "I couldn't find anything in this video that answers that.",
    "fr": "Je n'ai rien trouvé dans cette vidéo qui réponde à cette question.",
    "ar": "لم أجد في هذا الفيديو ما يجيب على هذا السؤال.",
}

EMBED_NOTICE = {
    "en": ("Search is running in keyword-only mode right now, so results may be "
           "less precise than usual."),
    "fr": ("La recherche fonctionne actuellement en mode mots-clés uniquement ; "
           "les résultats peuvent être moins précis que d'habitude."),
    "ar": ("البحث يعمل حالياً بالكلمات المفتاحية فقط، لذا قد تكون النتائج أقل دقة "
           "من المعتاد."),
}


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
            f"[{i}] time={fmt_ts(r.get('start_time'))} "
            f"(to {fmt_ts(r.get('end_time'))}) source={r.get('chunk_type')}\n{body}"
        )
    return "\n\n".join(parts)


SYSTEM = """You answer questions about a single video using only the numbered passages provided.

CRITICAL — output ONLY the final answer:
- Never show your reasoning, planning, or analysis.
- Never write phrases like "We need to", "Let's", "I should", "The user wants",
  "Thus main points", "Let me list", or any commentary about the passages.
- Never restate or summarise these instructions.
- Do not explain which passages you looked at. Just answer.
- Your entire reply must be the answer itself, starting with its first word.

OUTPUT FORMAT — follow exactly:
1. Open with one direct sentence that answers the question. No preamble, no restating the question.
2. Then a blank line, then numbered key points, one per line, in this shape:
   1. The point, stated plainly. (2:14) [3]
   2. The next point. (5:47) [1]
3. Give 2 to 5 numbered points. Skip the list entirely if one sentence fully answers it.

TIMESTAMP RULE — important:
- Write timestamps as M:SS or H:MM:SS in parentheses, e.g. (2:14) or (1:03:22).
- Take the time from the passage header you are citing. Every point that comes
  from a passage must carry its timestamp — the interface turns these into
  clickable links that jump the video to that moment.

CITATION RULE:
- Cite passages as [1], [2] after the claim they support.
- Never cite a passage you did not use.

STRICT — the interface shows plain text, so markdown is NOT rendered:
- Never use *, **, _, #, backticks, or bullet characters like - or •.
- Write "1." "2." "3." for the list. Nothing else.
- No headings, no bold, no code blocks.

GROUNDING:
- Use only the passages. If they do not answer the question, say so in one sentence and stop.
- Passages marked "visual" come from on-screen text and frame descriptions; passages marked "speech" come from the spoken transcript. Both are valid evidence.

LANGUAGE — the language to answer in is decided by the server and handed to you
explicitly as an "ANSWER LANGUAGE" line with the request. Obey that line exactly.
It overrides any impression you form from the passages, which are frequently in a
different language from the question: never answer in the passages' language when
the directive names another one.
- Keep technical terms in the form they naturally appear — usually English
  (deep learning, rank, matrix, gradient) or exactly as written on screen / spoken
  in the video. Do NOT translate technical terms into the answer language.
- Match the script too: answer an Arabic-script question in Arabic script, not
  transliteration."""


def strip_reasoning(txt: str) -> str:
    """
    Some free models emit chain-of-thought before the answer
    ("We need to answer... Let's list relevant passages... Thus main points:").
    Cut everything before the real answer begins.
    """
    import re as _re
    if not txt:
        return ""
    t = txt
    t = _re.sub(r"<think>.*?</think>", "", t, flags=_re.S | _re.I)
    t = _re.sub(r"<reasoning>.*?</reasoning>", "", t, flags=_re.S | _re.I)
    t = _re.sub(r"^.*?</think>", "", t, flags=_re.S | _re.I)

    for marker in ("Thus main points:", "Thus main points",
                   "Final answer:", "Answer:", "So the answer is:"):
        i = t.rfind(marker)
        if i != -1:
            t = t[i + len(marker):]
            break

    plan = _re.compile(
        r"^\s*(we (need|should|must|can)\b|let'?s\b|i (need|should|will|'ll)\b|"
        r"the user (wants|is asking)\b|okay,? (so|let)\b|first,? (i|we|let)\b|"
        r"now,? (i|we|let)\b|maybe\b|actually,? (i|we)\b)",
        _re.I)
    lines = t.split("\n")
    while lines and (not lines[0].strip() or plan.match(lines[0])):
        lines.pop(0)
    return "\n".join(lines).strip()


def clean_answer(txt: str) -> str:
    """
    The frontend renders answers as escaped plain text (api.js:525), so any
    markdown the model emits shows up literally as ** and ### on screen.
    Free models ignore formatting instructions fairly often, so strip it here.
    """
    import re
    if not txt:
        return ""
    t = txt
    t = re.sub(r"```[a-z]*\n?", "", t)          # code fences
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)  # headings
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)      # bold
    t = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", t)  # italics
    t = re.sub(r"(?<!\w)_(?!\s)(.+?)(?<!\s)_(?!\w)", r"\1", t)
    t = re.sub(r"`([^`]+)`", r"\1", t)          # inline code
    t = re.sub(r"^\s*[-*•]\s+", "", t, flags=re.M)  # bullet chars
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


async def generate_answer(question: str, context: str,
                          lang: str = "en") -> tuple[str, str]:
    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY not set")

    # Repeated before and after the passages on purpose: with a long context the
    # small free models reliably drift into the passages' language if the
    # directive only appears once, at the top.
    directive = language_directive(lang)
    user = (
        f"{directive}\n\n"
        f"Question: {question}\n\n"
        f"Passages from the video:\n\n{context}\n\n"
        f"Answer the question using only these passages, citing them as [n]. "
        f"Every claim must carry its [n] and the (M:SS) from the header of the "
        f"passage it came from \u2014 that is the evidence the interface links to."
        f"\n{directive}"
    )
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://askvgia.app",
        "X-Title": "AskVGIA",
    }

    errors: List[str] = []
    async with httpx.AsyncClient(timeout=120) as client:
        for attempt, model in enumerate(OPENROUTER_MODELS, 1):
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
                        "max_tokens": MAX_ANSWER_TOKENS,
                    },
                )
                if r.status_code >= 400:
                    errors.append(f"{model} -> {r.status_code}: {r.text[:180]}")
                    # A 429 comes in two flavours and they need opposite
                    # handling. "temporarily rate-limited upstream" is about one
                    # provider, so the next model is worth trying. The daily
                    # free-model quota is per ACCOUNT, so every remaining
                    # attempt would fail identically — walking the chain just
                    # adds latency and burns nothing useful. Stop and say so.
                    if r.status_code == 429 and "free-models-per-day" in r.text:
                        raise HTTPException(
                            429,
                            "The OpenRouter account has used up its free-model "
                            "quota for today. Answers resume when the quota "
                            "resets, or immediately if credits are added to the "
                            "account.")
                    continue
                data = r.json()
                if data.get("error"):
                    errors.append(f"{model} -> {str(data['error'])[:180]}")
                    continue

                choice = (data.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                # Reasoning models (most of what openrouter/free routes to) write
                # a "reasoning" field first and only then "content". If the token
                # budget runs out mid-thought the response is a valid HTTP 200
                # with content=null. The old code called .strip() straight on that
                # null, which threw AttributeError into the except below and fell
                # through to the next model — so a merely truncated answer looked
                # like a dead model, and the user got the last model's error.
                #
                # Deliberately NOT falling back to msg["reasoning"]: that field is
                # raw chain-of-thought ("Here's a thinking process: 1. Analyze
                # user input"), and strip_reasoning() cannot reliably tell it from
                # a real answer. Treat it as this model's failure and let the next
                # one answer instead. MAX_ANSWER_TOKENS is what keeps this rare.
                raw = msg.get("content") or ""
                if not raw.strip():
                    errors.append(
                        f"{model} -> empty content (finish_reason="
                        f"{choice.get('finish_reason')})")
                    continue

                txt = clean_answer(strip_reasoning(raw))
                if txt:
                    # data["model"] is the model the router actually picked, which
                    # is what should surface - not the "openrouter/free" alias.
                    return txt, data.get("model") or model
                errors.append(f"{model} -> answer empty after cleanup")
            except HTTPException:
                raise
            except Exception as e:
                errors.append(
                    f"attempt {attempt} {model} -> "
                    f"{type(e).__name__}: {str(e)[:180]}")
                continue
    # Report every model's failure, not just the last: in a fallback chain the
    # last error is usually the least informative one.
    raise HTTPException(502, "no free model answered. " + " | ".join(errors))


# ------------------------------------------------------------- derived data
# demo_videos only stores video_id, youtube_id, title, is_truncated, n_chunks.
# Every richer panel the frontend asks for (summary, keywords, topics, timeline,
# chapters, card thumbnails) is derived here from video_chunks and cached in
# process for the process lifetime. The corpus is static (ingestion is local),
# so a restart is the only invalidation needed. See DIAG.md for the full
# rationale behind each of these.
from collections import Counter as _Counter, OrderedDict as _OrderedDict


class _LRU(_OrderedDict):
    """Dict avec taille max. Évince l'entrée la moins récemment utilisée une
    fois plein — empêche la RAM de croître sans limite (Render free = 512MB)."""
    def __init__(self, maxsize=6):
        super().__init__()
        self.maxsize = maxsize

    def __getitem__(self, key):
        self.move_to_end(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        while len(self) > self.maxsize:
            self.popitem(last=False)


# le gros cache — listes de chunks complètes par vidéo ; plafond serré
_chunks_cache: "Dict[str, List[Dict[str, Any]]]" = _LRU(maxsize=4)
# petits caches (résumés / tags) ; on en garde un peu plus
_derived_cache: "Dict[str, Dict[str, Any]]" = _LRU(maxsize=20)
_tags_cache: "Dict[str, List[str]]" = _LRU(maxsize=20)

# youtube_id -> (title, retry_after_epoch). Successes never expire; a failed
# lookup is cached briefly so a YouTube hiccup can't add its timeout to every
# /videos call, but also can't leave the cards ugly until the next restart.
_yt_title_cache: Dict[str, tuple] = {}
_YT_TITLE_RETRY = 600     # seconds before re-trying a failed lookup


async def yt_title(youtube_id: str) -> str:
    """
    The real YouTube title, via the public oEmbed endpoint (no API key, no
    quota). demo_videos.title is hand-written and empty for the later videos,
    which left the cards showing the raw 11-char ID. Returns "" if YouTube is
    unreachable or the video is private/removed — callers fall back.
    """
    if not youtube_id:
        return ""
    import time
    hit = _yt_title_cache.get(youtube_id)
    if hit and (hit[0] or hit[1] > time.time()):
        return hit[0]
    title = ""
    try:
        async with httpx.AsyncClient(timeout=8) as client:
            r = await client.get(
                "https://www.youtube.com/oembed",
                params={"url": f"https://www.youtube.com/watch?v={youtube_id}",
                        "format": "json"},
            )
        if r.status_code < 400:
            title = (r.json().get("title") or "").strip()
    except Exception:
        title = ""
    _yt_title_cache[youtube_id] = (title, time.time() + _YT_TITLE_RETRY)
    return title

# tags that describe the recording rather than its subject — dropped from keywords
_TAG_STOP = {
    "video", "videos", "screenshot", "screen", "background", "dark background",
    "person", "people", "man", "woman", "blurry", "blur", "image", "photo",
    "picture", "frame", "footage", "camera", "closeup", "close-up", "indoor",
    "outdoor", "dark", "light", "text", "white", "black", "color", "colour",
}


async def fetch_chunks(video_id: str, select: str = "*",
                       extra: Optional[Dict[str, str]] = None,
                       limit: int = 5000) -> List[Dict[str, Any]]:
    """Read video_chunks for one video via PostgREST."""
    url = f"{SUPABASE_URL}/rest/v1/video_chunks"
    params = {"video_id": f"eq.{video_id}", "select": select, "limit": str(limit)}
    if extra:
        params.update(extra)
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.get(url, headers=_sb_headers(), params=params)
        return r.json() if r.status_code < 400 else []


async def all_chunks(video_id: str) -> List[Dict[str, Any]]:
    """
    All chunks for a video, time-ordered, cached. Paginated because PostgREST
    caps every response at 1000 rows (its `max-rows` setting) regardless of the
    `limit` we ask for — the MIT lecture has 1156 chunks, so a single read would
    silently drop the tail of the video from the timeline and chapters.
    """
    if video_id not in _chunks_cache:
        rows: List[Dict[str, Any]] = []
        page, offset = 1000, 0
        while True:
            batch = await fetch_chunks(
                video_id,
                select="chunk_id,start_time,end_time,text,topic,chunk_type,speaker,metadata",
                extra={"order": "start_time.asc", "offset": str(offset)},
                limit=page,
            )
            rows.extend(batch)
            if len(batch) < page:
                break
            offset += page
        _chunks_cache[video_id] = rows
    return _chunks_cache[video_id]


def _top_tags(rows: List[Dict[str, Any]], k: int) -> List[str]:
    c: _Counter = _Counter()
    for r in rows:
        if (r.get("chunk_type") or "") != "visual":
            continue
        for t in (r.get("metadata") or {}).get("tags") or []:
            t = str(t).strip().lower()
            if t and t not in _TAG_STOP and len(t) > 2:
                c[t] += 1
    return [t for t, _ in c.most_common(k)]


async def video_tags(video_id: str, k: int = 5) -> List[str]:
    """Top subject tags for a video card (sampled visual chunks), cached."""
    if video_id not in _tags_cache:
        rows = await fetch_chunks(
            video_id, select="chunk_type,metadata",
            extra={"chunk_type": "eq.visual"}, limit=150,
        )
        _tags_cache[video_id] = _top_tags(rows, 24)
    return _tags_cache[video_id][:k]


def _dedupe_speech(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    The store has OVERLAPPING speech chunks (sliding-window chunking), so the
    same sentence reappears at nearly the same timestamp inside a longer chunk.
    Exact-match dedup misses those — that is why the Timeline showed 00:09 twice.

    Drop a chunk when it starts within 12s of one already kept and its wording is
    largely contained in it. Keeps the fuller version.
    """
    kept: List[Dict[str, Any]] = []
    for r in sorted(rows, key=lambda x: float(x.get("start_time") or 0)):
        w = _norm(r.get("text") or "")
        st = float(r.get("start_time") or 0)
        if not w:
            continue
        dup_idx = None
        for i, prev in enumerate(kept):
            pw = _norm(prev.get("text") or "")
            if not pw or abs(st - float(prev.get("start_time") or 0)) > 12:
                continue
            if len(w & pw) / (min(len(w), len(pw)) or 1) >= 0.80:
                dup_idx = i
                break
        if dup_idx is None:
            kept.append(r)
        elif len(w) > len(_norm(kept[dup_idx].get("text") or "")):
            kept[dup_idx] = r
    return kept


async def derive_video(video_id: str) -> Dict[str, Any]:
    """summary / keywords / topics / duration_seconds from chunks, cached."""
    if video_id in _derived_cache:
        return _derived_cache[video_id]
    rows = await all_chunks(video_id)

    speech = sorted(
        (r for r in rows if (r.get("chunk_type") or "speech") != "visual"),
        key=lambda r: float(r.get("start_time") or 0),
    )
    speech = _dedupe_speech(speech)

    # summary — opening speech, or frame captions for a silent video
    summary = " ".join((r.get("text") or "").strip() for r in speech[:6]).strip()
    if not summary:
        caps: List[str] = []
        for r in rows:
            md = r.get("metadata") or {}
            cap = md.get("caption") or md.get("short_caption")
            if cap:
                caps.append(cap)
            if len(caps) >= 4:
                break
        summary = " ".join(caps)
    summary = summary[:600]

    keywords = _top_tags(rows, 12)

    tc: _Counter = _Counter()
    for r in rows:
        t = (r.get("topic") or "").strip()
        if t and t.lower() != "general":
            tc[t] += 1
    topics = [t for t, _ in tc.most_common(8)]

    duration = max((float(r.get("end_time") or 0) for r in rows), default=0.0)

    out = {"summary": summary, "keywords": keywords, "topics": topics,
           "duration_seconds": duration}
    _derived_cache[video_id] = out
    return out


def _yt_positional_thumb(yt: Optional[str], pos: float) -> Optional[str]:
    """
    YouTube auto-generates three preview frames per video, sampled at roughly
    25/50/75% of its length (mq1/mq2/mq3, 320x180). Pick the one nearest a
    chapter's position. Used only as a fallback: ingestion caps frame extraction
    (config/settings.py:131), so on a long video the tail chunks carry no
    frame_url and those chapters would otherwise render an empty icon.
    """
    if not yt:
        return None
    n = 1 if pos < 0.34 else (2 if pos < 0.67 else 3)
    return f"https://i.ytimg.com/vi/{yt}/mq{n}.jpg"


async def derive_chapters(video_id: str) -> List[Dict[str, Any]]:
    """
    Chapters from topic transitions in time order. Chunks with topic 'general'
    extend the open chapter instead of starting a meaningless one, and a new
    chapter only opens once the current one is at least `min_len` long — derived
    from duration (~one chapter per 2.5 min, 4–40) so long videos get a
    proportionally longer chapter list. See DIAG §5.
    """
    rows = await all_chunks(video_id)
    if not rows:
        return []
    duration = max((float(r.get("end_time") or 0) for r in rows), default=0.0)
    # ~one chapter per 2.5 min, floored at 4 and capped at 40, scaling to length.
    target_count = min(40, max(4, round(duration / 150.0)))
    min_len = max(30.0, duration / target_count)

    chapters: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    for r in rows:
        topic = (r.get("topic") or "").strip()
        st = float(r.get("start_time") or 0)
        en = float(r.get("end_time") or 0)
        md = r.get("metadata") or {}
        frame = md.get("frame_url")
        text = (r.get("text") or "").strip()
        is_general = (not topic) or topic.lower() == "general"

        start_new = (
            cur is None
            or (not is_general and topic != cur["topic"]
                and (cur["end_time"] - cur["start_time"]) >= min_len)
        )
        if start_new:
            cur = {
                "topic": topic if not is_general else (cur["topic"] if cur else "Intro"),
                "start_time": st, "end_time": en,
                "thumbnail_url": frame, "summary": text,
            }
            chapters.append(cur)
        else:
            cur["end_time"] = max(cur["end_time"], en)
            if not is_general and cur["topic"] in ("", "Intro"):
                cur["topic"] = topic
            if not cur.get("thumbnail_url") and frame:
                cur["thumbnail_url"] = frame
            if not cur.get("summary") and text:
                cur["summary"] = text

    # Only pay for the video lookup if some chapter actually needs the fallback.
    yt = None
    if any(not ch.get("thumbnail_url") for ch in chapters):
        yt = (await get_video(video_id)).get("youtube_id")

    out = []
    for i, ch in enumerate(chapters):
        pos = (ch["start_time"] / duration) if duration else 0.0
        out.append({
            "index": i,
            "title": (ch["topic"] or "Chapter").strip() or "Chapter",
            "start_time": ch["start_time"],
            "end_time": ch["end_time"],
            "summary": (ch.get("summary") or "")[:200],
            "thumbnail_url": ch.get("thumbnail_url") or _yt_positional_thumb(yt, pos),
        })
    return out


def filter_graph(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Drop edges whose endpoints aren't in the node set — d3.forceLink throws on
    the first dangling edge and aborts the whole render, blanking the panel.
    Then drop nodes left with no edges (the "scattered confetti" look), unless
    that empties the graph, in which case keep them so a sparse-but-real graph
    still draws instead of the "no data" placeholder. See DIAG §1.
    """
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    ids = {n.get("id") for n in nodes}
    good = [e for e in edges if e.get("source") in ids and e.get("target") in ids]
    connected = {e.get("source") for e in good} | {e.get("target") for e in good}
    trimmed = [n for n in nodes if n.get("id") in connected]
    if not trimmed:
        trimmed = nodes
    return {"nodes": trimmed, "edges": good}


# ----------------------------------------------------------------- routes
@app.get("/health")
async def health():
    return {
        "ok": True,
        "supabase": bool(SUPABASE_ANON_KEY),
        "hf": bool(HF_TOKEN),
        "openrouter": bool(OPENROUTER_API_KEY),
    }


# --- background warmer ----------------------------------------------------
# /warmup is a monitor ping, so it has to answer in milliseconds. Embedding
# inline made the response time equal to HF latency — measured at 1s, 3s and
# 22s within the same minute — and embed_question's retry ladder can hold a
# request open for over two minutes on a bad HF stretch. That is longer than
# any timeout UptimeRobot offers, so the monitor opened a false "down"
# incident while the service was perfectly healthy. The ping now schedules the
# warm and returns; hitting the route is what keeps Render awake either way.
_warm_state: Dict[str, Any] = {"running": False, "last_ok": None, "note": "", "at": 0.0}
_warm_task: Optional[asyncio.Task] = None


async def _do_warm() -> None:
    import time
    try:
        await embed_question("warmup")
        _warm_state.update(last_ok=True, note="", at=time.time())
    except Exception as e:
        _warm_state.update(last_ok=False, note=str(e)[:120], at=time.time())
    finally:
        _warm_state["running"] = False


# HEAD as well as GET: UptimeRobot sends HEAD by default (it is the lighter
# request, and choosing the method is a paid feature), and FastAPI's @app.get —
# unlike Starlette's plain Route — does NOT register HEAD alongside GET. So a
# GET-only route answers every check with 405 and the monitor sits permanently
# down on a healthy service.
@app.api_route("/warmup", methods=["GET", "HEAD"])
async def warmup():
    """Ping target for UptimeRobot. Kicks off a background embed so the HF
    BGE-M3 model stays warm, preventing the cold-start 'model is warming up' on
    a juror's first question. Returns immediately and never raises, so the
    monitor measures this service's latency rather than HuggingFace's."""
    global _warm_task
    import time
    if not _warm_state["running"]:
        _warm_state["running"] = True
        # Keep the reference: a bare create_task can be garbage-collected mid-flight.
        _warm_task = asyncio.create_task(_do_warm())
    return {
        "ok": True,
        "warming": _warm_state["running"],
        "last_ok": _warm_state["last_ok"],
        "age_s": round(time.time() - _warm_state["at"], 1) if _warm_state["at"] else None,
        "note": _warm_state["note"] or None,
    }


# --- history (ask_history, per-user + 7-day RLS from 05_history_peruser.sql) --
# These four routes are the only ones that talk to Supabase as the CALLER rather
# than as the anon role: they forward the browser's `Authorization: Bearer <jwt>`
# straight through. The anon key resolves to auth.uid() = NULL, so the per-user
# RLS policy would match no rows at all. With the user's token, Postgres runs the
# query as that user, which is what scopes history to them and hides rows older
# than 7 days — there is deliberately no user_id filter or date cutoff here.
def _user_headers(authorization: str) -> Dict[str, str]:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": authorization,
        "Content-Type": "application/json",
    }


def _jwt_sub(authorization: str) -> Optional[str]:
    """
    The `sub` claim (the user's uuid) out of a bearer JWT, WITHOUT verifying it.

    Only used to fill user_id on insert, because the RLS insert check requires
    user_id = auth.uid(). Reading it unverified is safe: Postgres verifies the
    signature on the same token and rejects the row if the two disagree, so a
    forged sub buys nothing. Avoids pulling in a JWT library for one field.
    """
    try:
        payload = authorization.split(" ", 1)[1].split(".")[1]
        payload += "=" * (-len(payload) % 4)          # base64url, padding stripped
        return json.loads(base64.urlsafe_b64decode(payload)).get("sub")
    except Exception:
        return None


@app.get("/history")
async def history_list(authorization: Optional[str] = Header(None)):
    if not authorization:
        return []                        # guest: no history, not an error
    url = f"{SUPABASE_URL}/rest/v1/ask_history"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(url, headers=_user_headers(authorization),
                                 params={"select": "*", "order": "created_at.desc",
                                         "limit": "100"})
            return r.json() if r.status_code < 400 else []
    except Exception:
        return []


@app.post("/history")
async def history_add(item: Dict[str, Any],
                      authorization: Optional[str] = Header(None)):
    if not authorization:
        return {"ok": True}
    uid = _jwt_sub(authorization)
    if not uid:
        return {"ok": True}
    # Built field by field: a client-supplied user_id is never trusted, and any
    # other key the frontend sends would be a PostgREST 400 (unknown column).
    row = {
        "user_id": uid,
        "video_id": item.get("video_id") or item.get("videoId"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "metadata": item.get("metadata") or {},
    }
    url = f"{SUPABASE_URL}/rest/v1/ask_history"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            await client.post(url, headers=_user_headers(authorization), json=row)
    except Exception:
        pass
    return {"ok": True}


@app.delete("/history/{item_id}")
async def history_del(item_id: str, authorization: Optional[str] = Header(None)):
    if not authorization:
        return {"ok": True}
    url = f"{SUPABASE_URL}/rest/v1/ask_history"
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            await client.delete(url, headers=_user_headers(authorization),
                                params={"id": f"eq.{item_id}"})
    except Exception:
        pass
    return {"ok": True}


@app.delete("/history")
async def history_clear(authorization: Optional[str] = Header(None)):
    if not authorization:
        return {"ok": True}
    url = f"{SUPABASE_URL}/rest/v1/ask_history"
    try:
        # RLS narrows this to the caller's own rows, so "every id" is safe.
        async with httpx.AsyncClient(timeout=20) as client:
            await client.delete(url, headers=_user_headers(authorization),
                                params={"id": "not.is.null"})
    except Exception:
        pass
    return {"ok": True}


# --- serve the existing frontend unchanged -------------------------------
# Same origin, so api.js's relative paths (/ask, /videos) just work.
from pathlib import Path as _P
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

_APP = _P("static/app")
_SITE = _P("project")

if (_APP / "css").exists():
    app.mount("/app-assets", StaticFiles(directory=str(_APP)), name="app-assets")

# The marketing pages (project/*.html) reference images as a RELATIVE
# "assets/..." path, which resolves to /assets/... once they are served from
# the site root — so that directory needs its own mount or every logo and
# icon on the landing, login, signup and pricing pages 404s.
if (_SITE / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_SITE / "assets")), name="site-assets")


@app.get("/app/{page:path}")
async def app_page(page: str):
    p = _APP / (page or "index.html")
    if p.is_dir() or not page:
        p = _APP / "index.html"
    if not p.exists():
        p = _APP / "index.html"
    return FileResponse(str(p))


@app.get("/")
async def root():
    for c in (_SITE / "VGIA.html", _SITE / "index.html", _APP / "index.html"):
        if c.exists():
            return FileResponse(str(c))
    return {"ok": True, "hint": "frontend files not found"}


@app.get("/{page}.html")
async def site_page(page: str):
    p = _SITE / f"{page}.html"
    if not p.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(p))


@app.get("/videos")
async def videos():
    """Shape matches the existing frontend: { videos: [...] } (api.js d.videos)."""
    url = f"{SUPABASE_URL}/rest/v1/demo_videos_ext"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_sb_headers(),
                             params={"select": "*", "order": "n_chunks.desc"})
        if r.status_code >= 400:
            raise HTTPException(502, r.text[:200])
        rows = r.json()

    # Top subject tags + the real YouTube title per card, fetched concurrently
    # (both cached after the first call).
    tag_lists, yt_titles = await asyncio.gather(
        asyncio.gather(*[video_tags(v.get("video_id")) for v in rows],
                       return_exceptions=True),
        asyncio.gather(*[yt_title(v.get("youtube_id")) for v in rows],
                       return_exceptions=True),
    )

    out = []
    for v, tags, ytt in zip(rows, tag_lists, yt_titles):
        if isinstance(tags, Exception):
            tags = []
        if isinstance(ytt, Exception):
            ytt = ""
        yt = v.get("youtube_id")
        n = v.get("n_chunks", 0)
        out.append({
            **v,
            "id": v.get("video_id"),
            # Real YouTube title first, hand-written title as the fallback.
            "title": ytt or v.get("title") or v.get("video_id"),
            "status": "ready",
            "indexed": True,
            "total_chunks": n,           # the frontend reads this (api.js:564,1066)
            "chunk_count": n,
            "summary": " · ".join(tags) if tags else "",   # keyword line on the card
            "keywords": tags,
            # Both keys on purpose: the cards read `thumbnail_url`, older code
            # reads `thumbnail`. Keeping them in sync avoids a silent blank card.
            "thumbnail": f"https://i.ytimg.com/vi/{yt}/mqdefault.jpg" if yt else None,
            "thumbnail_url": f"https://i.ytimg.com/vi/{yt}/mqdefault.jpg" if yt else None,
            "youtube_url": f"https://www.youtube.com/watch?v={yt}" if yt else None,
        })
    return {"videos": out, "total": len(out)}


@app.get("/video/{video_id}/chapters")
async def video_chapters(video_id: str):
    """Chapters derived from topic transitions (api.js:753)."""
    return {"chapters": await derive_chapters(video_id)}


@app.get("/video/{video_id}")
async def video_detail(video_id: str):
    v = await get_video(video_id)
    if not v:
        raise HTTPException(404, "video not found")
    yt = v.get("youtube_id")
    derived = await derive_video(video_id)   # summary / keywords / topics / duration
    n = v.get("n_chunks", 0)
    ytt = await yt_title(yt)                 # real YouTube title, cached
    return {**v, **derived, "title": ytt or v.get("title") or v.get("video_id"),
            "id": v.get("video_id"), "status": "ready",
            "indexed": True, "total_chunks": n, "chunk_count": n,
            "thumbnail": f"https://i.ytimg.com/vi/{yt}/mqdefault.jpg" if yt else None,
            "thumbnail_url": f"https://i.ytimg.com/vi/{yt}/mqdefault.jpg" if yt else None}


@app.post("/search")
async def search(req: Dict[str, Any]):
    """
    Timeline tab (api.js:716). It wants speech chunks in TIME order, not
    relevance order — it lists timestamps — so this deliberately does not run
    the hybrid search. Deduplicated because the store has overlapping chunks.
    """
    video_id = req.get("video_id")
    if not video_id:
        raise HTTPException(400, "video_id required")
    rows = await all_chunks(video_id)
    # ALL speech chunks in time order — the Timeline shows the full video, not a
    # top-k. De-duplicated because the store has overlapping chunks.
    speech = _dedupe_speech(
        sorted((r for r in rows if (r.get("chunk_type") or "speech") != "visual"),
               key=lambda r: float(r.get("start_time") or 0))
    )
    results = []
    for r in speech:
        st = float(r.get("start_time") or 0)
        en = float(r.get("end_time") or 0)
        results.append({
            "chunk_id": r.get("chunk_id"),
            "text": r.get("text") or "",
            "content": r.get("text") or "",
            "start_time": st, "end_time": en, "start": st, "end": en,
            "chunk_type": r.get("chunk_type") or "speech",
            "speaker": r.get("speaker"),
            "metadata": r.get("metadata") or {},
        })
    return {"results": results, "chunks": results, "total": len(results)}


@app.get("/vision/frames/{video_id}")
async def vision_frames(video_id: str, limit: int = 60):
    """Visual chunks that carry a stored frame image (api.js:1050 thumbnails)."""
    rows = await fetch_chunks(
        video_id, select="chunk_id,start_time,end_time,metadata",
        extra={"chunk_type": "eq.visual", "order": "start_time.asc"},
        limit=max(limit, 8),
    )
    frames = []
    for r in rows:
        fu = (r.get("metadata") or {}).get("frame_url")
        if not fu:
            continue
        frames.append({
            "chunk_id": r.get("chunk_id"), "frame_url": fu,
            "start_time": r.get("start_time"), "end_time": r.get("end_time"),
        })
        if len(frames) >= limit:
            break
    return {"frames": frames, "total": len(frames)}


@app.get("/vision/frame/{chunk_id}")
async def vision_frame(chunk_id: str):
    """Redirect to the stored frame image for a chunk (api.js:393,1054)."""
    url = f"{SUPABASE_URL}/rest/v1/video_chunks"
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(url, headers=_sb_headers(),
                             params={"chunk_id": f"eq.{chunk_id}",
                                     "select": "metadata", "limit": "1"})
        rows = r.json() if r.status_code < 400 else []
    fu = (rows[0].get("metadata") or {}).get("frame_url") if rows else None
    if not fu:
        raise HTTPException(404, "frame not found")
    return RedirectResponse(fu)


@app.get("/graph/{video_id}")
@app.get("/graph/video/{video_id}")
async def graph(video_id: str):
    url = f"{SUPABASE_URL}/rest/v1/rpc/get_video_graph"
    async with httpx.AsyncClient(timeout=45) as client:
        r = await client.post(url, json={"p_video_id": video_id}, headers=_sb_headers())
        if r.status_code >= 400:
            raise HTTPException(502, r.text[:200])
        data = r.json()
    return filter_graph(data if isinstance(data, dict) else {})


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    q = (req.question or "").strip()
    if not q:
        raise HTTPException(400, "question is empty")

    video = await get_video(req.video_id)
    youtube_id = video.get("youtube_id")

    # Answer language: an explicit client choice wins, otherwise detect it from
    # the question. req.language used to be accepted and silently dropped, which
    # is why picking a language in the UI changed nothing.
    requested = (req.language or "").strip().lower()[:2]
    if requested in LANG_NAMES:
        lang, lang_source = requested, "requested"
    else:
        lang, lang_source = detect_language(q), "detected"

    # Dense retrieval when we can, keyword-only when the embedder is down.
    notice = None
    retrieval_mode = "hybrid"
    try:
        vec = await embed_question(q)
        # over-fetch so dedup still leaves TOP_K distinct passages
        rows = await hybrid_search(req.video_id, q, vec, limit=TOP_K * 3)
    except EmbeddingUnavailable as e:
        retrieval_mode = "keyword"
        notice = EMBED_NOTICE.get(lang, EMBED_NOTICE["en"])
        print(f"[ask] embeddings unavailable (transient={e.transient}): "
              f"{e.detail} - falling back to keyword search")
        rows = await keyword_search(req.video_id, q, limit=TOP_K * 3)

    rows = dedupe_visual(rows)[:TOP_K]

    if not rows:
        return AskResponse(
            answer=NO_MATCH.get(lang, NO_MATCH["en"]),
            citations=[], video_id=req.video_id, youtube_id=youtube_id,
            language=lang, language_source=lang_source,
            retrieval_mode=retrieval_mode, notice=notice,
        )

    context = build_context(rows)
    answer, model_used = await generate_answer(q, context, lang=lang)

    cites = []
    for r in rows:
        md = r.get("metadata") or {}
        st = float(r.get("start_time") or 0)
        en = float(r.get("end_time") or 0)
        cites.append(Citation(
            chunk_id=r.get("chunk_id", ""),
            text=(r.get("text") or "")[:400],
            start_time=st,
            end_time=en,
            start=st,
            end=en,
            score=float(r.get("rrf_score") or 0),
            video_id=r.get("video_id"),
            metadata=md,
            chunk_type=r.get("chunk_type") or "speech",
            speaker=r.get("speaker"),
            frame_url=md.get("frame_url"),
            youtube_url=(f"https://www.youtube.com/watch?v={youtube_id}&t={int(st)}s"
                         if youtube_id else None),
        ))

    return AskResponse(
        answer=answer, citations=cites, chunks=cites, sources=cites, results=cites,
        video_id=req.video_id, youtube_id=youtube_id, model_used=model_used,
        language=lang, language_source=lang_source,
        retrieval_mode=retrieval_mode, notice=notice,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)
