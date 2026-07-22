from __future__ import annotations


import asyncio
import hashlib
import json
import logging
import mimetypes
import os
import re
import shutil
import time
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── logging ──────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api.main")


# =====================================================================
#  DotDict — dict qui supporte aussi getattr (pour AgenticSearch)
# =====================================================================

class DotDict(dict):
    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            return None

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            pass


# =====================================================================
# HARD IMPORTS
# =====================================================================

from core.storage import VideoStore
from core.transcription import VideoTranscriber
from core.chunking import SemanticChunker
from core.embeddings import EmbeddingGenerator
from core.reranker import CrossEncoderReranker
from core.vision_indexer import VisionIndexer, VisualChunk
from core.vlm_contracts import VLMConfig
from core.clip_embedder import CLIPEmbedder
from core.vqa_engine import VQAEngine, VisualStoreAdapter
from core import llm_client
from core.metadata import VideoMetadata, VideoMetadataRegistry, get_registry
from core.query_cache import QueryCache
from core.auth import get_auth_db, public_user as _public_user_dict, AuthDB
from config.settings import settings

logger.info("✅ All core imports OK")

try:
    from core.search import AgenticSearch
    _AGENTIC_AVAILABLE = True
    logger.info("✅ AgenticSearch available")
except ImportError as _e:
    AgenticSearch = None
    _AGENTIC_AVAILABLE = False
    logger.info("ℹ️  AgenticSearch not available (%s)", _e)

try:
    from core.graph_store import GraphStore
    from core.graph_extractor import GraphExtractor
    from core.graph_builder import GraphBuilder
    from core.graph_search import GraphSearch
    GRAPHRAG_AVAILABLE = True
    logger.info("✅ GraphRAG modules OK")
except ImportError as _e:
    GraphStore = None
    GraphExtractor = None
    GraphBuilder = None
    GraphSearch = None
    GRAPHRAG_AVAILABLE = False
    logger.info("ℹ️  GraphRAG not available (%s)", _e)

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = BASE_DIR / "data" / "videos"
STATIC_DIR  = BASE_DIR / "static"
PROJECT_DIR = BASE_DIR / "project"   # marketing site (landing, login, signup, pricing, docs)
DATA_DIR.mkdir(parents=True, exist_ok=True)

_ingest_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")
_search_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="search")

app = FastAPI(title="Video Intelligence API", version="6.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── No-cache middleware for JS/CSS/HTML assets ───────────────────────
# Prevents browsers from serving stale JS after server-side updates.
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

class NoCacheAssetsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.endswith((".js", ".css", ".html")):
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response

app.add_middleware(NoCacheAssetsMiddleware)

APP_DIR = STATIC_DIR / "app"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Serve entire app subfolder as /app-assets so HTML pages can reference css/js
if APP_DIR.exists():
    app.mount("/app-assets", StaticFiles(directory=str(APP_DIR)), name="app_assets")

# ── singletons ───────────────────────────────────────────────────────
transcriber: Optional[Any] = None
chunker:     Optional[Any] = None
embedder:    Optional[Any] = None
reranker:    Optional[Any] = None
vision:      Optional[Any] = None
store:       Optional[Any] = None
agentic:     Optional[Any] = None

graph_store:     Optional[Any] = None
graph_extractor: Optional[Any] = None
graph_builder:   Optional[Any] = None
graph_search:    Optional[Any] = None

vqa_engine: Optional[Any] = None
_frame_index: Optional[Any] = None

# BUG-4: video-level metadata registry
_metadata_registry: Optional[VideoMetadataRegistry] = None

# CACHE: semantic query cache
_query_cache: Optional[QueryCache] = None

# AUTH: user/session database
_auth_db: Optional[AuthDB] = None


# =====================================================================
# STARTUP HEALTH CHECK
# =====================================================================

_CRITICAL_COMPONENTS = ["store", "embedder", "transcriber", "chunker"]


def _assert_components_ready() -> None:
    missing = [name for name in _CRITICAL_COMPONENTS if globals().get(name) is None]
    if missing:
        logger.critical(
            "\n═══ STARTUP FAILED ═══\n"
            "  Critical components not initialised: %s\n"
            "  Check logs above for the exact error and restart.",
            ", ".join(missing),
        )
        sys.exit(1)
    logger.info("✅ Health check passed — all critical components ready")


def _init_components() -> None:
    global transcriber, chunker, embedder, reranker, vision, store, agentic
    global graph_store, graph_extractor, graph_builder, graph_search
    global vqa_engine, _frame_index, _metadata_registry, _query_cache, _auth_db

    def _safe_init(name, cls, target):
        if cls is None:
            return None
        try:
            logger.info("⏳ Starting %s ...", name)  # ← AJOUTE
            t0 = time.time()  # ← AJOUTE
            obj = cls()
            logger.info("✅ %s ready (%.1fs)", name, time.time() - t0)  # ← MODIFIE
            return obj
        except Exception as e:
            logger.warning("⚠️  %s init failed: %s", name, e)
            return None

    if transcriber is None:
        transcriber = _safe_init("VideoTranscriber", VideoTranscriber, "transcriber")
    if chunker is None:
        chunker = _safe_init("SemanticChunker", SemanticChunker, "chunker")
    if embedder is None:
        embedder = _safe_init("EmbeddingGenerator", EmbeddingGenerator, "embedder")
    if reranker is None:
        import concurrent.futures as _cf
        logger.info("⏳ Starting CrossEncoderReranker (timeout=30s) ...")
        with _cf.ThreadPoolExecutor(max_workers=1) as _ex:
            _fut = _ex.submit(CrossEncoderReranker)
            try:
                reranker = _fut.result(timeout=30)
                logger.info("✅ CrossEncoderReranker ready")
            except _cf.TimeoutError:
                logger.warning("⚠️  CrossEncoderReranker TIMEOUT — reranker disabled")
                reranker = None
            except Exception as e:
                logger.warning("⚠️  CrossEncoderReranker failed: %s", e)
                reranker = None

    if vision is None and VisionIndexer is not None:
        logger.info("⏳ Starting VisionIndexer ...")  # ← AJOUTE
        t0 = time.time()  # ← AJOUTE
        try:
            vision = VisionIndexer(settings=settings)
            logger.info("✅ VisionIndexer ready (%.1fs) VLM=%s",
                        time.time() - t0,
                        vision.vlm.backend_name if vision.vlm else "none")
        except Exception as e:
            logger.warning("⚠️  VisionIndexer init failed: %s", e)

    # IMPORTANT: store MUST be initialised BEFORE _frame_index,
    # because VisualStoreAdapter wraps the shared VideoStore.
    if store is None:
        store = _safe_init("VideoStore", VideoStore, "store")

    # BUG-2 FIX: use VisualStoreAdapter wrapping the shared VideoStore.
    # Also pass vision indexer so CLIP FAISS index is used for search.
    if _frame_index is None and store is not None:
        try:
            _frame_index = VisualStoreAdapter(store, vision_indexer=vision)
            logger.info("✅ Visual frame index ready (VisualStoreAdapter → CLIP FAISS + VideoStore)")
        except Exception as e:
            logger.warning("Frame index init failed: %s", e)
            _frame_index = None

    if (
        vqa_engine is None
        and VQAEngine is not None
        and vision is not None
        and _frame_index is not None            # GUARD: never create VQAEngine without a frame index
    ):
        try:
            if vision.clip is not None and vision.vlm is not None:
                vqa_engine = VQAEngine(
                    vlm=vision.vlm,
                    clip=vision.clip,
                    frame_index=_frame_index,
                    top_k_retrieve=12,
                    top_n_answer=4,
                )
                logger.info(
                    "✅ VQA engine ready (VLM=%s, CLIP=%s, frame_index=VisualStoreAdapter)",
                    vision.vlm.backend_name,
                    vision.clip.backend_name,
                )
            else:
                logger.warning("VQA engine not created: CLIP or VLM missing")
        except Exception as e:
            logger.warning("VQA engine init failed: %s", e)
            vqa_engine = None

    # BUG-4: metadata registry
    if _metadata_registry is None:
        try:
            persist_dir = getattr(settings, "storage_dir", "data/faiss_store")
            _metadata_registry = get_registry(persist_dir=persist_dir)
            logger.info("✅ VideoMetadataRegistry ready (%d records)", len(_metadata_registry))
        except Exception as e:
            logger.warning("⚠️  VideoMetadataRegistry init failed: %s", e)

    # CACHE: semantic query cache (initialised after embedder is ready)
    if _query_cache is None:
        try:
            persist_dir = getattr(settings, "storage_dir", "data/faiss_store")
            _query_cache = QueryCache(
                storage_dir=persist_dir,
                embedder=embedder,   # shares the same EmbeddingGenerator
            )
            logger.info("✅ QueryCache ready")
        except Exception as e:
            logger.warning("⚠️  QueryCache init failed: %s", e)

    if agentic is None and AgenticSearch is not None:
        try:
            agentic = AgenticSearch(
                store=store,
                embedder=embedder,
                reranker=reranker,
                llm_client=llm_client,
            )
            logger.info("✅ AgenticSearch ready")
        except Exception as e:
            logger.warning("⚠️  AgenticSearch init failed: %s", e)

    # AUTH DB
    if _auth_db is None:
        try:
            persist_dir = getattr(settings, "storage_dir", "data/faiss_store")
            avatars_dir = str(BASE_DIR / "data" / "avatars")
            _auth_db = get_auth_db(storage_dir=persist_dir, avatars_dir=avatars_dir)
            logger.info("✅ AuthDB ready")
        except Exception as e:
            logger.warning("⚠️  AuthDB init failed: %s", e)

    if not GRAPHRAG_AVAILABLE:
        logger.info("GraphRAG skipped (modules not importable)")
        return

    if graph_store is None and GraphStore is not None:
        try:
            graph_persist_dir = getattr(settings, "graph_store_dir", str(BASE_DIR / "data" / "graph_store"))
            graph_store = GraphStore(persist_dir=str(graph_persist_dir))
            logger.info("✅ GraphStore ready (nodes=%d)",
                        graph_store.graph.number_of_nodes())
        except Exception as e:
            logger.error("GraphStore init failed: %s", e, exc_info=True)
            graph_store = None

    if graph_extractor is None and GraphExtractor is not None:
        try:
            from core.graph_extractor import ExtractorConfig as _EC
            _ecfg = _EC(
                max_workers=getattr(settings, "graph_extractor_max_workers", 1),
                entity_timeout_s=getattr(settings, "graph_extract_timeout_s", 45),
                relation_timeout_s=getattr(settings, "graph_extract_relation_timeout_s", 60),
                max_text_chars=getattr(settings, "graph_extract_max_text_chars", 900),
                entity_num_predict=getattr(settings, "graph_extract_num_predict", 300),
                relation_num_predict=getattr(settings, "graph_extract_num_predict", 300),
                max_entities_per_chunk=getattr(settings, "graph_max_entities_per_chunk", 10),
                max_relations_per_chunk=getattr(settings, "graph_max_relations_per_chunk", 8),
                use_heuristic_fallback=getattr(settings, "graph_extract_use_heuristic_fallback", True),
            )
            graph_extractor = GraphExtractor(llm_client=llm_client, config=_ecfg)
            logger.info("✅ GraphExtractor ready")
        except Exception as e:
            logger.warning("GraphExtractor init failed: %s", e)
            graph_extractor = None

    if (
        graph_builder is None
        and GraphBuilder is not None
        and graph_store is not None
        and graph_extractor is not None
    ):
        try:
            graph_builder = GraphBuilder(
                store=graph_store,
                extractor=graph_extractor,
                llm_client=llm_client,
            )
            logger.info("✅ GraphBuilder ready")
        except Exception as e:
            logger.warning("GraphBuilder init failed: %s", e)
            graph_builder = None

    if graph_search is None and GraphSearch is not None and graph_store is not None:
        try:
            graph_search = GraphSearch(
                graph_store=graph_store,
                vector_store=store,
                embedder=embedder,
                reranker=reranker,
                llm_client=llm_client,
            )
            logger.info("✅ GraphSearch ready")
        except Exception as e:
            logger.warning("GraphSearch init failed: %s", e)
            graph_search = None


@app.on_event("startup")
async def startup() -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _init_components)
    _assert_components_ready()
    logger.info("════════════════════════════════════════")
    logger.info("  Video Intelligence API v6.2 — ready")
    logger.info("════════════════════════════════════════")


# =====================================================================
# ── BUG-7: Pydantic models (updated) ─────────────────────────────────
# =====================================================================

# ─────────────────────────────────────────────────────────────────────
# BENCHMARK MODE — all answers use this dataset-tuned prompt unless a request
# explicitly overrides it with its own "benchmark" field.
#   "video_mmlu" : short/exact math answers (recall few words, why/how 1 sentence)
#   "eduvidqa"   : thorough reasoned paragraphs
#   None         : normal default prompt
# To switch benchmarks later, change this ONE line.
# ─────────────────────────────────────────────────────────────────────
DEFAULT_BENCHMARK = "video_mmlu"

# Per-video Q&A log (question + answer + grounding) written live as you ask.
# Grouped by video_id; multiple questions accumulate under the same video.
QA_LOG_PATH = os.environ.get("QA_LOG_PATH", "qa_log.json")


class AskRequest(BaseModel):
    question: str
    video_id: Optional[str] = None
    # v7.0: confirmed=True by default — skip clarification gate for direct answers
    confirmed: bool = True
    reformulated_question: Optional[str] = None
    # existing fields
    llm: bool = True
    vision: bool = False
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    language_hint: Optional[str] = None
    use_cache: bool = False  # STEP-1: semantic cache disabled by default (was True).
                             # Re-enable per-request with use_cache=true if desired.
    # UNIFIED PIPELINE: graph enrichment baked into the streaming endpoint
    use_graph: bool = True   # inject graph context into LLM synthesis
    graph_mode: str = "auto" # auto | local | hybrid | global
    benchmark: Optional[str] = None  # 'video_mmlu' | 'eduvidqa' | None — per-request override of DEFAULT_BENCHMARK


class ClarificationResponse(BaseModel):
    """
    BUG-7: Returned by /ask when confirmed=False.
    The frontend shows clarification + two buttons: confirm or rephrase.
    """
    clarification: str        # human-readable "Do you mean: …?"
    reformulated: str         # cleaner version of the question
    intent: str               # factual|summary|temporal|visual|speaker|entity
    entities: List[str]       # key entities detected in the question
    confirmed: bool = False   # always False in this response


class SearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: int = 10
    with_vision: bool = False


class IngestRequest(BaseModel):
    video_path: str
    video_id: Optional[str] = None
    language: Optional[str] = None


# ── helpers ──────────────────────────────────────────────────────────

def _cfg(name: str, default: Any = None) -> Any:
    if settings is None:
        return default
    return getattr(settings, name, default)


_YES_NO_START_RE = re.compile(r"^(did|does|do|is|are|was|were|can|could|will|has|have)\b", re.IGNORECASE)

def _is_stale_cache_answer(question: str, cached: dict) -> bool:
    """
    Return True if a cached answer looks like a stale bad answer that should
    be regenerated instead of served:
      - Answer starts with 'YES' but the question is not a yes/no question
        (catches the old YES-bleed bug).
    """
    answer: str = (
        cached.get("answer")
        or (cached.get("llm") or {}).get("answer")
        or ""
    )
    if answer.startswith("YES") and not _YES_NO_START_RE.match(question.strip()):
        logger.info("Cache stale (YES-bleed): skipping cached answer for %r", question[:60])
        return True
    return False


def _generate_video_id(filename: str) -> str:
    h = hashlib.md5(filename.encode("utf-8")).hexdigest()[:8]
    safe = "".join(c if c.isalnum() else "_" for c in Path(filename).stem)[:40]
    return f"{h}_{safe}"


def _find_video_file(video_id: str) -> Optional[Path]:
    if not video_id or not DATA_DIR.exists():
        return None
    for f in DATA_DIR.iterdir():
        if f.is_file() and video_id in f.name:
            return f
    for f in DATA_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in (
            ".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".m4v",
        ):
            if _generate_video_id(f.name) == video_id:
                return f
    return None


def _format_ts(seconds: Any) -> str:
    try:
        s = max(0.0, float(seconds))
    except Exception:
        return "00:00"
    total = int(round(s))
    h, remainder = divmod(total, 3600)
    m, sec = divmod(remainder, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def _resolve_search_top_k(requested_top_k: Optional[int], *, with_vision: bool = False) -> int:
    """Candidate pool for reranking. Keep small — reranker is CPU-bound and slow.
    Vector search already ranked well; reranker just needs the top candidates.

    VISUAL-FIX: Long videos can have hundreds of visual frames spread across the
    index. The relevant frame may not be in the top-6 by embedding score alone
    (frame captions are short; semantic drift is higher than for speech chunks).
    We therefore use a larger candidate pool when vision is involved so the
    reranker has enough material to surface the correct frame.
    """
    if requested_top_k is not None:
        return max(1, min(int(requested_top_k), 50))
    # Speech-only: 6 candidates is enough — transcription chunks are long and
    # semantically rich, so the embedding rank is very reliable.
    # Vision-included: use 16 so long-video visual frames are not cut off before
    # the reranker gets a chance to score them.
    return 16 if with_vision else 6


def _resolve_rerank_top_k(requested_top_k: Optional[int], candidate_count: int) -> int:
    if candidate_count <= 0:
        return 0
    if requested_top_k is None:
        # Hard cap at 6: reranker is CPU-bound (~3s/chunk on CPU).
        # Reranking 6 chunks ≈ 8-10s. Quality is fine — vector search pre-ranks well.
        return min(6, candidate_count)
    return max(1, min(int(requested_top_k), candidate_count))


def _clean_visual_text(raw):
    """Remove raw JSON/dict/metadata leakage from visual chunk text fields."""
    import re as _re
    if not raw:
        return raw
    s = raw.strip()
    has_text_key = ("'text'" in s) or ('"text"' in s)
    if s.startswith(('{', '[')) or has_text_key:
        # quoted value: 'text': 'Hello world'  or  "text": "Hello"
        m = _re.search(r"""['"](text)['"]\s*:\s*['"]([^'"]{4,})['"]""", s)
        if m:
            return m.group(2).strip()
        # unquoted: text: Hello world authority: 0.0
        m = _re.search(r"""\btext\b\s*:\s*(.+?)(?:\s+\w+\s*[:{]|$)""", s, _re.IGNORECASE)
        if m:
            cand = m.group(1).strip().strip("'\"")
            if len(cand) > 4:
                return cand
        return ""
    # Strip trailing dict noise appended after real text
    s = _re.sub(r'\s*\{[^}]{0,200}\}\s*$', '', s).strip()
    # Strip trailing metadata key:value pairs
    s = _re.sub(
        r'\s+(authority|confidence|language|score|confidence_level)\s*:\s*\S+.*$',
        '', s, flags=_re.IGNORECASE
    ).strip()
    return s


def _normalize_result(r: Any) -> DotDict:
    FIELDS = (
        "chunk_id", "video_id", "text", "original_text", "search_text",
        "score", "rerank_score", "dense_score", "sparse_score",
        "start_time", "end_time", "language", "speaker",
        "chunk_type", "key_points", "metadata", "strategy",
        # visual-specific fields that may be top-level
        "caption", "short_caption", "ocr_text", "description", "label",
    )
    d = DotDict()
    if isinstance(r, dict):
        for k in FIELDS:
            if k in r:
                d[k] = r[k]
        for k, v in r.items():
            if k not in d:
                d[k] = v
    else:
        for k in FIELDS:
            val = getattr(r, k, None)
            if val is not None:
                d[k] = val
    if not d.get("text"):
        d["text"] = d.get("original_text") or d.get("search_text") or ""
    # FIX: For visual chunks, build a display_text from all possible caption
    # sources so the frontend always has human-readable text to show.
    # Priority: short_caption → caption → ocr_text → description → text
    if (d.get("chunk_type") or "speech") == "visual":
        meta = d.get("metadata") or {}
        if isinstance(meta, str):
            try:
                import json as _json
                meta = _json.loads(meta)
            except Exception:
                meta = {}
        raw_candidates = [
            d.get("short_caption") or meta.get("short_caption"),
            d.get("caption")       or meta.get("caption"),
            d.get("ocr_text")      or meta.get("ocr_text"),
            d.get("description")   or meta.get("description"),
            d.get("label")         or meta.get("label"),
            d.get("text"),
        ]
        # Clean each candidate to strip raw JSON/metadata leakage
        cleaned = [_clean_visual_text(str(c)) if c else "" for c in raw_candidates]
        caption_text = next((c for c in cleaned if c and c.strip()), "")

        # VERBATIM FIX: board_text / screen_text hold the EXACT on-screen content
        # (formulas, code, algorithms, bullet lists) transcribed by the VLM. These
        # were NOT in raw_candidates, so display_text only ever carried the caption
        # ("the sigmoid outputs 0..1") and the actual formula was dropped before the
        # LLM ever saw it. Prepend the verbatim on-screen text so questions like
        # "give me the formula / algorithm / code shown" can be answered exactly.
        # NOTE: do NOT pass board_text through _clean_visual_text — its trailing-brace
        # strip would truncate formulas like "g'(z) = {1, if z>0; 0 otherwise}".
        def _safe_verbatim(v: Any) -> str:
            s = str(v or "").strip()
            if not s or s.upper() in ("BLANK", "NONE"):
                return ""
            # only treat as leakage if it's clearly a serialized dict/record
            if s.startswith(("{'", '{"')) and ("'text'" in s or '"text"' in s):
                return _clean_visual_text(s) or ""
            return s

        board = _safe_verbatim(d.get("board_text") or meta.get("board_text"))
        screen = _safe_verbatim(d.get("screen_text") or meta.get("screen_text"))
        verbatim_parts: List[str] = []
        if board:
            verbatim_parts.append("On-screen text:\n" + board)
        # only add screen_text if it adds something not already in board_text
        if screen and screen not in board:
            verbatim_parts.append("Other on-screen text: " + screen)

        if verbatim_parts:
            verbatim_block = "\n".join(verbatim_parts)
            # put the exact on-screen text FIRST (it's the highest-value evidence
            # for educational formula/code/algorithm questions), then the caption
            d["display_text"] = (
                verbatim_block + (("\n" + caption_text) if caption_text else "")
            )
        else:
            d["display_text"] = caption_text
    d["start_fmt"] = _format_ts(d.get("start_time"))
    d["end_fmt"]   = _format_ts(d.get("end_time"))
    return d


def _video_is_already_indexed(video_id: str) -> bool:
    if store is None:
        return False
    try:
        stats = store.get_stats(video_id=video_id)
        if isinstance(stats, dict):
            count = stats.get("total_chunks") or stats.get("chunks") or 0
            return int(count) > 0
        return False
    except Exception:
        return False


def _call(obj: Any, method_names: list, *args, **kwargs) -> Any:
    for name in method_names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(*args, **kwargs)
    raise AttributeError(f"{type(obj).__name__} has none of {method_names}")


# =====================================================================
# BUG-7 FIX: Clarification logic
# =====================================================================

_CLARIFICATION_PROMPT = """\
You are a video search assistant. The user asked:
'{question}'
Respond ONLY with valid JSON, no other text:
{{
  "reformulated": "cleaner, specific version of the question",
  "intent": "factual|summary|temporal|visual|speaker|entity",
  "entities": ["list", "of", "key", "entities"],
  "clarification": "Do you mean: [reformulated]? Are you asking about [entities]?"
}}
If the question is already clear and specific, set reformulated = original question.\
"""

_AMBIGUITY_INDICATORS = (
    # Pronouns without clear referent
    " it ", " he ", " she ", " they ", " this ", " that ", " those ", " these ",
    "tell me about it",
    "what did he say",
    "what did she say",
    "what did they say",
    "what about that",
    "the thing he",
    "the thing she",
    "the part where",
    "that part",
    "this part",
    "what was that",
    "you know",
)


def _is_ambiguous(question: str) -> bool:
    """
    Heuristic check: is the question likely to produce low-quality answers
    due to unresolved pronoun references or vague phrasing?
    """
    q_lower = question.lower().strip()

    # Very short questions are usually ambiguous
    word_count = len(q_lower.split())
    if word_count <= 3:
        return True

    for indicator in _AMBIGUITY_INDICATORS:
        if indicator in f" {q_lower} ":
            return True

    return False


def _run_clarification_sync(question: str) -> ClarificationResponse:
    """
    BUG-7 FIX: Runs a single fast JSON-mode LLM call (150 tokens, temp=0)
    to produce a clarification/reformulation of the question.
    Executed in _search_executor.
    """
    prompt = _CLARIFICATION_PROMPT.format(question=question)

    raw_json: Optional[str] = None

    # Try llm_client first (it knows the configured model)
    if llm_client is not None:
        try:
            # Many llm_client implementations expose a raw_complete / generate method
            for method_name in ["raw_complete", "complete", "generate", "chat"]:
                fn = getattr(llm_client, method_name, None)
                if callable(fn):
                    result = fn(
                        prompt=prompt,
                        max_tokens=150,
                        temperature=0.0,
                        json_mode=True,
                    )
                    if isinstance(result, str):
                        raw_json = result
                    elif isinstance(result, dict):
                        raw_json = result.get("text") or result.get("content") or result.get("response")
                    break
        except Exception as e:
            logger.warning("llm_client clarification call failed: %s", e)

    # Fallback: direct Ollama HTTP call
    if raw_json is None:
        try:
            import urllib.request
            ollama_url = getattr(settings, "ollama_base_url", "http://localhost:11434")
            model = getattr(settings, "ollama_chat_model", "qwen3.5:2b")
            payload = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 150,
                },
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw_json = body.get("response", "")
        except Exception as e:
            logger.warning("Ollama clarification fallback failed: %s", e)

    # Parse JSON response
    if raw_json:
        try:
            # Strip markdown code fences if present
            cleaned = raw_json.strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join(
                    line for line in cleaned.split("\n")
                    if not line.strip().startswith("```")
                ).strip()
            parsed = json.loads(cleaned)
            return ClarificationResponse(
                clarification=parsed.get(
                    "clarification",
                    f"Do you mean: {parsed.get('reformulated', question)}?",
                ),
                reformulated=parsed.get("reformulated", question),
                intent=parsed.get("intent", "factual"),
                entities=parsed.get("entities", []),
                confirmed=False,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Failed to parse clarification JSON: %s — raw: %s", e, raw_json[:200])

    # Graceful degradation: return the original question without blocking
    logger.info("Clarification unavailable — returning original question as reformulated")
    return ClarificationResponse(
        clarification=f'Do you mean: "{question}"?',
        reformulated=question,
        intent="factual",
        entities=[],
        confirmed=False,
    )


# ── ingestion pipeline ────────────────────────────────────────────────

def _run_ingest_sync(
    video_path: str,
    video_id: str,
    language: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.time()
    result: Dict[str, Any] = {
        "video_id": video_id,
        "video_path": video_path,
        "status": "pending",
        "steps": [],
    }

    if _video_is_already_indexed(video_id):
        logger.info("⏭️  video_id=%s already indexed — skip", video_id)
        result["status"] = "skipped"
        result["reason"] = "already_indexed"
        return result

    logger.info("▶ ingestion START video_id=%s path=%s", video_id, video_path)

    # 1) Transcription
    transcription_result = None
    segments: list = []
    if transcriber is not None:
        try:
            transcription_result = _call(
                transcriber,
                ["process_video", "transcribe"],
                video_path,
                language=language,
            )
            segments = (
                getattr(transcription_result, "segments", None)
                or getattr(transcription_result, "results", None)
                or (transcription_result if isinstance(transcription_result, list) else [])
            )
            result["steps"].append(f"transcription: {len(segments)} segments")
            logger.info("  transcription → %d segments", len(segments))
        except Exception as e:
            logger.error("  transcription FAILED: %s", e)
            result["steps"].append(f"transcription: FAILED — {e}")
    else:
        result["steps"].append("transcription: SKIPPED")

    # 2) Chunking
    chunks: list = []
    video_summary = None

    if chunker is not None and segments:
        try:
            chunk_result = _call(
                chunker,
                ["chunk", "process", "create_chunks"],
                segments,
                video_id=video_id,
            )
            chunks = (
                    getattr(chunk_result, "chunks", None)
                    or (chunk_result if isinstance(chunk_result, list) else [])
            )
            video_summary = getattr(chunk_result, "video_summary", None)

            result["steps"].append(f"chunking: {len(chunks)} chunks")
            if video_summary:
                result["steps"].append(
                    f"summary ready: domain={getattr(video_summary, 'domain', '?')}"
                )
            logger.info("  chunking → %d chunks (summary=%s)",
                        len(chunks), "yes" if video_summary else "no")

        except Exception as e:
            logger.error("  chunking FAILED: %s", e, exc_info=True)
            result["steps"].append(f"chunking: FAILED — {e}")
            # chunks reste [], video_summary reste None
    else:
        result["steps"].append("chunking: SKIPPED")
    # ── GPU HANDOFF ──────────────────────────────────────────────────────
    # WhisperX (~3.0 GB) stays in VRAM after transcription. With it loaded,
    # qwen3-vl:8b only gets ~4.5 GB headroom → Ollama spills to CPU →
    # ~25 s/frame instead of ~3-4 s/frame. Releasing here fixes that.
    # 7-min video: was ~60 min VLM, becomes ~8 min after this change.
    if transcriber is not None:
        _rel = getattr(transcriber, "release_gpu", None)
        if callable(_rel):
            try:
                _rel()
                logger.info("ingest: WhisperX VRAM released — VLM has full GPU")
            except Exception as _e:
                logger.warning("ingest: release_gpu failed (non-critical): %s", _e)
        else:
            # Fallback for older transcription.py without release_gpu()
            try:
                import torch as _t, gc as _g
                _wm = getattr(transcriber, "whisper_model", None)
                if _wm is not None:
                    del transcriber.whisper_model
                    transcriber.whisper_model = None
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
                _g.collect()
                logger.info("ingest: manual WhisperX VRAM release done")
            except Exception as _e2:
                logger.warning("ingest: manual VRAM release failed: %s", _e2)

    # 3) Vision indexing
    visual_chunks: list = []
    if vision is not None and hasattr(vision, 'enabled') and vision.enabled:
        try:
            t_v = time.time()
            visual_chunks = vision.index_video(
                video_path=video_path,
                video_id=video_id,
                language_hint=language,
                extract_tags=True,
                extract_ocr=True,
                save_json=True,
            )
            result["steps"].append(f"vision: {len(visual_chunks)} chunks")
            result["visual_chunks"] = len(visual_chunks)
            if hasattr(vision, 'vlm') and vision.vlm:
                result["vlm_backend"] = vision.vlm.backend_name
                result["vlm_model"] = vision.vlm.model_name
            logger.info("  vision OK in %.1fs (%d chunks)", time.time() - t_v, len(visual_chunks))
        except Exception as e:
            logger.exception("vision pipeline failed: %s", e)
            result["steps"].append(f"vision: FAILED ({e})")
    else:
        result["steps"].append("vision: SKIPPED (disabled or unavailable)")

    # BUG-2 FIX: visual chunks are ALREADY persisted in the shared VideoStore
    # via store.add_embedded_chunks(...) in step 5 below.
    # VisualStoreAdapter reads them back directly — no separate frame index to feed.

    all_chunks = list(chunks) + list(visual_chunks)

    # ── DESCRIPTION CHUNK: index the per-video narrative description so global-
    # property questions (background color, presenting academy, overall style)
    # have something to retrieve. Without this, "what color is the background"
    # returns zero results because that fact lives in the description, not in any
    # per-frame chunk. The description is generated by vision_indexer into
    # self._video_descriptions[video_id].
    try:
        _desc = ""
        if vision is not None and hasattr(vision, "_video_descriptions"):
            _desc = (vision._video_descriptions or {}).get(video_id, "") or ""
        if _desc.strip():
            _desc_chunk = VisualChunk(
                chunk_id=f"{video_id}__description",
                video_id=video_id,
                text=_desc.strip(),
                timestamp=0.0,
                end_timestamp=0.0,
                start_time=0.0,
                chunk_type="video_description",
            )
            all_chunks.append(_desc_chunk)
            result["steps"].append("description: indexed as 1 chunk")
            logger.info("  description chunk added (%d chars)", len(_desc))
    except Exception as _de:
        logger.warning("description chunk indexing failed (ignored): %s", _de)

    # 4) Embedding
    embedded: list = []
    if embedder is not None and all_chunks:
        try:
            test_embedding = embedder.embed_query("test")
            if test_embedding and test_embedding.bge_dense:
                logger.info(f"✅ BGE-M3 OK: dense vector length = {len(test_embedding.bge_dense)}")
            else:
                logger.error("❌ BGE-M3 returned empty dense vector! This will cause storage failure.")
        except Exception as e:
            logger.error(f"❌ BGE-M3 test failed: {e}")

        try:
            embed_result = _call(
                embedder,
                ["generate_chunks", "embed", "embed_chunks", "process"],
                all_chunks,
                video_id=video_id,
            )
            embedded = (
                getattr(embed_result, "chunks", None)
                or (embed_result if isinstance(embed_result, list) else [])
            )
            result["steps"].append(f"embedding: {len(embedded)} embedded")
            logger.info("  embedding → %d embedded chunks", len(embedded))
        except Exception as e:
            logger.error("  embedding FAILED: %s", e)
            result["steps"].append(f"embedding: FAILED — {e}")
    else:
        result["steps"].append("embedding: SKIPPED")

    # 5) Storage
    if store is not None and embedded:
        try:
            # Persistance forcée : sans cela, une vidéo avec < snapshot_interval_chunks
            # peut disparaître après refresh / restart avant le prochain snapshot auto.
            store.add_embedded_chunks(embedded, save=True)
            result["steps"].append(f"storage: {len(embedded)} chunks stored + persisted")
            logger.info("  storage → %d chunks stored and persisted", len(embedded))
        except TypeError:
            # Compatibilité avec une ancienne implémentation du store.
            store.add_embedded_chunks(chunks, save=True, prewarm_bm25=True)

            if hasattr(store, "save"):
                store.save()
            result["steps"].append(f"storage: {len(embedded)} chunks stored + persisted")
            logger.info("  storage → %d chunks stored and persisted", len(embedded))
        except Exception as e:
            logger.error("  storage FAILED: %s", e)
            result["steps"].append(f"storage: FAILED — {e}")
    else:
        result["steps"].append("storage: SKIPPED")

    # 6) Graph building — use all_chunks (speech + visual + description) so
    # entity extraction has rich content even for near-silent videos.
    _graph_chunks = all_chunks if all_chunks else chunks
    if graph_builder is not None and _graph_chunks:
        try:
            gres = graph_builder.build_video_graph(
                video_id=video_id,
                chunks=_graph_chunks,
                save=True,
            )
            result["steps"].append(
                f"graph: {gres.n_nodes} nodes / {gres.n_edges} edges / "
                f"{gres.n_communities} communities"
            )
            result["graph"] = gres.as_dict()
            logger.info(
                "  graph → %d nodes, %d edges, %d communities (%.1fs)",
                gres.n_nodes, gres.n_edges, gres.n_communities, gres.elapsed_seconds,
            )
            # ── SPEAKER ENRICHMENT: inject speaker/scene nodes from metadata ──
            # This runs WITHOUT Ollama — pure metadata scan, always succeeds.
            if graph_store is not None and store is not None:
                try:
                    enrich_stats = graph_store.enrich_from_video_store(
                        video_store=store, video_id=video_id,
                    )
                    graph_store.save()
                    logger.info(
                        "  enrich_from_video_store → speakers=%d scenes=%d new_nodes=%d",
                        enrich_stats.get("speakers", 0),
                        enrich_stats.get("scenes", 0),
                        enrich_stats.get("new_nodes", 0),
                    )
                    result["steps"].append(
                        f"graph_enrich: speakers={enrich_stats.get('speakers',0)} "
                        f"scenes={enrich_stats.get('scenes',0)}"
                    )
                except Exception as _enrich_err:
                    logger.warning("  enrich_from_video_store failed: %s", _enrich_err)
        except Exception as e:
            logger.warning("  graph FAILED: %s", e)
            result["steps"].append(f"graph: FAILED — {e}")
    else:
        result["steps"].append("graph: SKIPPED")
        # Even without full graph building, enrich speaker nodes from metadata
        if graph_store is not None and store is not None:
            try:
                enrich_stats = graph_store.enrich_from_video_store(
                    video_store=store, video_id=video_id,
                )
                graph_store.save()
                logger.info(
                    "  enrich_from_video_store (no graph_builder) → speakers=%d",
                    enrich_stats.get("speakers", 0),
                )
            except Exception as _enrich_err:
                logger.warning("  enrich_from_video_store (fallback) failed: %s", _enrich_err)

    # ── 7) BUG-4: Generate & persist video-level metadata ──────────────
    if _metadata_registry is not None:
        try:
            # Detect language from chunks
            dominant_lang = language or "unknown"
            if chunks:
                langs = [getattr(c, "language", None) or
                         (c.get("language") if isinstance(c, dict) else None)
                         for c in chunks[:20]]
                langs = [l for l in langs if l and l != "unknown"]
                if langs:
                    from collections import Counter
                    dominant_lang = Counter(langs).most_common(1)[0][0]

            # Detect duration from last chunk
            dur: Optional[float] = None
            if chunks:
                last = chunks[-1]
                end_t = getattr(last, "end_time", None) or (last.get("end_time") if isinstance(last, dict) else None)
                if end_t:
                    try:
                        dur = float(end_t)
                    except Exception:
                        pass

            meta = _metadata_registry.generate_and_register(
                video_id=video_id,
                chunks=chunks,            # text chunks only (richer content)
                video_summary=video_summary,
                title=Path(video_path).stem.replace("_", " ").replace("-", " "),
                language=dominant_lang,
                duration_seconds=dur,
                total_chunks=len(embedded) if embedded else len(all_chunks),
                has_visual=len(visual_chunks) > 0,
                file_path=video_path,
                llm_client=llm_client,
                ollama_url=getattr(settings, "ollama_base_url", "http://localhost:11434"),
                ollama_model=getattr(settings, "ollama_chat_model", "qwen3.5:2b"),
            )
            result["steps"].append(
                f"metadata: title={meta.title!r} keywords={meta.keywords[:3]}"
            )
            result["metadata"] = meta.to_dict()
            logger.info(
                "  metadata → title=%r lang=%s topics=%s",
                meta.title, meta.language, meta.topics,
            )
        except Exception as e:
            logger.warning("  metadata generation FAILED: %s", e)
            result["steps"].append(f"metadata: FAILED — {e}")
    else:
        result["steps"].append("metadata: SKIPPED (registry not available)")

    # ── 8) CHAPTER-FIX: Pre-build chapters and cache in metadata.extra ──
    # This runs synchronously at the end of ingest (all chunks are stored).
    # After this, /video/{id} returns chapters instantly — no extra API call.
    if _metadata_registry is not None and store is not None:
        try:
            _ch_meta = _metadata_registry.get(video_id)
            if _ch_meta is not None and "chapters" not in _ch_meta.extra:
                # Collect speech chunks
                _fn = getattr(store, "get_all_chunks_for_video", None)
                _raw_ch: list = []
                if callable(_fn):
                    _raw_ch = _fn(video_id=video_id, speech_only=False) or []
                _speech_ch = []
                for _c in _raw_ch:
                    _ct = (_c.get("chunk_type") if isinstance(_c, dict) else getattr(_c, "chunk_type", "speech")) or "speech"
                    if _ct == "visual":
                        continue
                    _st = float((_c.get("start_time") if isinstance(_c, dict) else getattr(_c, "start_time", 0)) or 0)
                    _et = float((_c.get("end_time") if isinstance(_c, dict) else getattr(_c, "end_time", _st + 30)) or _st + 30)
                    _tx = (_c.get("text") or _c.get("content")) if isinstance(_c, dict) else (getattr(_c, "text", "") or getattr(_c, "content", ""))
                    _speech_ch.append({"start": _st, "end": _et, "text": (_tx or "").strip()})
                _speech_ch.sort(key=lambda x: x["start"])
                if _speech_ch:
                    _WINDOW = 60.0
                    _groups: list = []
                    _cur = None
                    for _chunk in _speech_ch:
                        if _cur is None or _chunk["start"] >= _cur["start"] + _WINDOW:
                            if _cur:
                                _groups.append(_cur)
                            _cur = {"start": _chunk["start"], "end": _chunk["end"], "texts": [_chunk["text"]]}
                        else:
                            _cur["end"] = max(_cur["end"], _chunk["end"])
                            if _chunk["text"]:
                                _cur["texts"].append(_chunk["text"])
                    if _cur:
                        _groups.append(_cur)
                    _meta_topics = _ch_meta.topics or _ch_meta.keywords or []
                    _built_chapters = []
                    for _i, _g in enumerate(_groups):
                        _topic = _meta_topics[_i] if _i < len(_meta_topics) else None
                        _combined = " ".join(_g["texts"])
                        if not _topic:
                            _topic = _combined[:60].rstrip() + ("…" if len(_combined) > 60 else "")
                        _summary = _combined[:200].rstrip() + ("…" if len(_combined) > 200 else "")
                        _built_chapters.append({
                            "index": _i + 1,
                            "title": _topic or f"Chapter {_i + 1}",
                            "start_time": _g["start"],
                            "end_time": _g["end"],
                            "summary": _summary,
                            "thumbnail_url": None,  # frames may not be ready yet; /chapters endpoint fills this
                        })
                    if _built_chapters:
                        _ch_meta.extra["chapters"] = _built_chapters
                        _ch_meta.touch()
                        _metadata_registry.register(_ch_meta)
                        result["steps"].append(f"chapters: pre-built {len(_built_chapters)} chapters")
                        logger.info("  chapters → pre-built %d chapters and cached in metadata", len(_built_chapters))
        except Exception as _ch_err:
            logger.warning("  chapter pre-build failed (non-fatal): %s", _ch_err)

    elapsed = time.time() - started
    result["status"] = "completed"
    result["elapsed_seconds"] = round(elapsed, 2)
    logger.info(
        "✅ ingestion DONE video_id=%s in %.1fs — %s",
        video_id, elapsed, " → ".join(result["steps"]),
    )
    return result


# ── search pipeline ──────────────────────────────────────────────────

def _strip_translate_prefix(query: str) -> str:
    """For translation requests strip the meta-directive so we embed the topic,
    not the instruction (e.g. 'translate the summary in arabic' → 'this video')."""
    import re as _re
    _prefix = _re.compile(
        r"^\s*(?:can\s+you\s+|please\s+)?(?:translat\w*|traduction|ترجم(?:ة)?)\s+"
        r"(?:the\s+|la\s+|le\s+|)?(?:summary\s+of\s+|résumé\s+de\s+|ملخص\s+)?",
        _re.IGNORECASE,
    )
    _suffix = _re.compile(
        r"\s+(?:in(?:to)?\s+)?(?:arabic|french|english|arabe|français|anglais"
        r"|العربية|الفرنسية|الإنجليزية)\s*\??\s*$",
        _re.IGNORECASE,
    )
    stripped = _prefix.sub("", query)
    stripped = _suffix.sub("", stripped).strip(" ?!")
    return stripped if len(stripped) >= 3 else query


# ── FIX: STT/name aliases — keep multiple surface forms together so both
# retrieval and final answering understand common transcription variants.
_STT_NAME_EQUIVALENTS: List[List[str]] = [
    ["Yoshua Bengio", "Joshua Bengio", "Yoshua Benjio", "Joshua Benjio"],
    # "Jeffington" is a known STT corruption of "Geoffrey Hinton" / "Jeff Hinton"
    ["Geoffrey Hinton", "Jeffrey Hinton", "Jeffington", "Jeff Hinton",
     "Geoffry Hinton", "Geoffry", "Geoff Hinton"],
    ["Yann LeCun", "Yan LeCun", "Yann Le Cun", "Yan Le Cun"],
]

def _expand_query_aliases(query: str) -> str:
    """Append equivalent name spellings when one variant is present.

    Example:
        'did they mention Geoffrey Hinton' ->
        'did they mention Geoffrey Hinton Jeffrey Hinton'
    """
    base = (query or "").strip()
    if not base:
        return query

    extras: List[str] = []
    lowered = base.lower()
    for variants in _STT_NAME_EQUIVALENTS:
        matched = any(v.lower() in lowered for v in variants)
        if not matched:
            continue
        for variant in variants:
            if variant.lower() not in lowered and variant not in extras:
                extras.append(variant)

    if not extras:
        return query
    return f"{base} {' '.join(extras)}"


def _do_search_sync(
    query: str,
    video_id: Optional[str] = None,
    top_k: Optional[int] = None,
    with_vision: bool = False,
) -> List[DotDict]:
    if store is None or embedder is None:
        return []

    search_query = _strip_translate_prefix(query)
    if not search_query or not search_query.strip():
        logger.warning("_do_search_sync: empty query after stripping, skipping search.")
        return []

    search_query = _expand_query_aliases(search_query)

    from core.llm_client import (
        _detect_intent as _detect_query_intent,
        _extract_all_timestamps as _extract_all_ts,
        _filter_results_by_timestamp as _filter_by_ts,
        classify_query_modality as _classify_modality,
    )
    import re as _re

    query_intent = _detect_query_intent(query or search_query)
    q_text = (query or search_query or "").lower()
    target_timestamps = _extract_all_ts(query or search_query)

    # ── MODALITY CLASSIFICATION ───────────────────────────────────
    # LLM-powered: understands what kind of evidence the query needs.
    # Fast regex covers 90% of cases (0 ms); LLM only for ambiguous ones.
    # Returns: 'speech' | 'visual' | 'both'
    _modality = _classify_modality(query or search_query, use_llm=False)  # STEP-1: keyword rules, no LLM call; always-multimodal
    logger.info("query modality=%s | q=%r", _modality, (query or search_query)[:80])

    speech_timestamp_query = (
        query_intent == "timestamp_lookup"
        and bool(_re.search(
            r"\b(say|said|says|talk|talked|talks|talking|mention|mentioned|speak|spoke|words?|quote|discuss|discussed|about)\b",
            q_text,
        ))
    )
    # visual_query now driven by LLM classifier, not keyword regex
    visual_query = (_modality == "visual" or query_intent == "visual")
    semantic_with_vision = bool(
        with_vision
        or _modality in ("visual", "both")
        or (query_intent == "timestamp_lookup" and not speech_timestamp_query)
    )

    def _chunk_type_of(item: Any) -> str:
        return str(getattr(item, "chunk_type", None) or (item.get("chunk_type") if isinstance(item, dict) else None) or "speech")

    def _result_id(item: Any) -> str:
        return str(getattr(item, "chunk_id", None) or (item.get("chunk_id") if isinstance(item, dict) else None) or "")

    def _result_start(item: Any) -> float:
        try:
            return float(getattr(item, "start_time", None) or (item.get("start_time") if isinstance(item, dict) else None) or getattr(item, "start", None) or (item.get("start") if isinstance(item, dict) else None) or 0.0)
        except Exception:
            return 0.0

    def _result_end(item: Any) -> float:
        try:
            end = float(getattr(item, "end_time", None) or (item.get("end_time") if isinstance(item, dict) else None) or getattr(item, "end", None) or (item.get("end") if isinstance(item, dict) else None) or _result_start(item))
        except Exception:
            end = _result_start(item)
        return max(_result_start(item), end)

    def _result_score(item: Any) -> float:
        for key in ("rerank_score", "score", "dense_score", "sparse_score"):
            val = getattr(item, key, None) or (item.get(key) if isinstance(item, dict) else None)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    pass
        return 0.0

    def _dedupe(items: List[Any]) -> List[DotDict]:
        merged: List[DotDict] = []
        seen: set = set()
        for raw in items:
            item = _normalize_result(raw)
            rid = _result_id(item) or f"anon:{len(merged)}:{_result_start(item):.3f}"
            if rid in seen:
                continue
            seen.add(rid)
            merged.append(item)
        return merged

    def _filter_modality(items: List[DotDict]) -> List[DotDict]:
        if not items:
            return []
        if speech_timestamp_query:
            speech_only = [item for item in items if _chunk_type_of(item) != "visual"]
            return speech_only if speech_only else items
        if visual_query:
            visual_only = [item for item in items if _chunk_type_of(item) == "visual"]
            return visual_only if visual_only else items
        return items

    def _filter_any_timestamp(items: List[DotDict], window: float) -> List[DotDict]:
        if not target_timestamps:
            return items
        merged: List[DotDict] = []
        seen: set = set()
        for ts in target_timestamps:
            for item in _filter_by_ts(items, ts, window=window):
                rid = _result_id(item) or f"anon:{_result_start(item):.3f}:{_result_end(item):.3f}"
                if rid in seen:
                    continue
                seen.add(rid)
                merged.append(item)
        return merged or items

    def _timestamp_sort_key(item: DotDict):
        start = _result_start(item)
        end = _result_end(item)
        contains = any(start <= ts <= end for ts in target_timestamps) if target_timestamps else False
        distance = 0.0
        if target_timestamps:
            distance = min(
                0.0 if start <= ts <= end else min(abs(start - ts), abs(end - ts))
                for ts in target_timestamps
            )
        modality_penalty = 0
        chunk_type = _chunk_type_of(item)
        if speech_timestamp_query and chunk_type == "visual":
            modality_penalty = 1
        if visual_query and chunk_type != "visual":
            modality_penalty = 1
        duration_penalty = max(0.0, end - start)
        return (
            0 if contains else 1,
            modality_penalty,
            round(distance, 6),
            round(duration_penalty, 6),
            -_result_score(item),
        )

    def _merge_exact_time_hits(current: List[DotDict]) -> List[DotDict]:
        if not target_timestamps or not video_id or not hasattr(store, "get_chunks_by_time_range"):
            return current
        exact_hits: List[DotDict] = []
        # FIX: tight windows first — only expand if nothing found
        windows = (5.0, 12.0, 20.0) if speech_timestamp_query else (8.0, 15.0, 25.0)
        for window in windows:
            batch: List[Any] = []
            for ts in target_timestamps:
                try:
                    batch.extend(store.get_chunks_by_time_range(
                        video_id=video_id,
                        start_seconds=max(0.0, ts - window),
                        end_seconds=ts + window,
                    ) or [])
                except Exception:
                    continue
            hits = _filter_modality(_filter_any_timestamp(_dedupe(batch), window))
            if hits:
                exact_hits = hits
                break  # found at tight window — do NOT expand further
        if not exact_hits:
            return current
        return _dedupe(exact_hits + current)

    def _direct_timestamp_search() -> List[DotDict]:
        """Bypass vector search for timestamp queries — go straight to time-range lookup."""
        if not target_timestamps or not video_id or not hasattr(store, "get_chunks_by_time_range"):
            return []
        results: List[DotDict] = []
        for ts in target_timestamps:
            for window in (5.0, 10.0, 20.0, 35.0):
                try:
                    batch = store.get_chunks_by_time_range(
                        video_id=video_id,
                        start_seconds=max(0.0, ts - window),
                        end_seconds=ts + window,
                    ) or []
                except Exception:
                    break
                hits = _filter_modality(_filter_any_timestamp(_dedupe(batch), window))
                if hits:
                    results.extend(hits)
                    break
        return _dedupe(results)

    try:
        query_emb = _call(
            embedder,
            ["embed_query", "generate_query", "encode_query"],
            search_query,
        )

        candidate_top_k = _resolve_search_top_k(top_k, with_vision=semantic_with_vision)
        # VISUAL-FIX: For long videos, visual frames are sparse in the index.
        # Use at least 8 visual candidates (not candidate_top_k // 2) so we
        # don't miss relevant frames that rank lower by embedding score alone.
        visual_top_k = max(8, candidate_top_k)

        # FIX: For timestamp queries with a specific video, use direct time-range
        # lookup FIRST — vector search finds semantically similar text which may
        # come from a completely different moment in the video.
        if target_timestamps and video_id and query_intent == "timestamp_lookup":
            direct_hits = _direct_timestamp_search()
            if direct_hits:
                results = direct_hits
                results = _merge_exact_time_hits(results)
                if target_timestamps:
                    results = sorted(results, key=_timestamp_sort_key)
                return results[:max(4, top_k or 6)]

        raw_results = store.search(
            query_embedding=query_emb,
            query=search_query,
            video_id=video_id,
            top_k=candidate_top_k,
            with_vision=semantic_with_vision,
        )

        visual_only: List[Any] = []
        if semantic_with_vision:
            try:
                visual_only = store.search(
                    query_embedding=query_emb,
                    query=search_query,
                    video_id=video_id,
                    top_k=visual_top_k,
                    with_vision=True,
                    chunk_type="visual",
                )
            except TypeError:
                _all_visual = store.search(
                    query_embedding=query_emb,
                    query=search_query,
                    video_id=video_id,
                    top_k=visual_top_k * 3,
                    with_vision=True,
                )
                visual_only = [
                    r for r in (_all_visual or [])
                    if (getattr(r, "chunk_type", None) or (r.get("chunk_type") if isinstance(r, dict) else None) or "speech") == "visual"
                ][:visual_top_k]
            except Exception:
                visual_only = []

        if not raw_results and not visual_only:
            return []

        results = _dedupe(list(raw_results or []) + list(visual_only or []))
        results = _merge_exact_time_hits(results)
        results = _filter_modality(results)

        if target_timestamps:
            tight_window = 18.0 if speech_timestamp_query else 28.0
            results = _filter_any_timestamp(results, tight_window)
            results = _filter_modality(results)
            results = sorted(results, key=_timestamp_sort_key)
        elif visual_query:
            visuals = [item for item in results if _chunk_type_of(item) == "visual"]
            non_visuals = [item for item in results if _chunk_type_of(item) != "visual"]
            if visuals:
                results = visuals + non_visuals

        should_rerank = reranker is not None and results and query_intent != "timestamp_lookup"
        if should_rerank:
            try:
                rerank_top_k = _resolve_rerank_top_k(top_k, len(results))
                reranked = _call(
                    reranker,
                    ["rerank", "rerank_results"],
                    query,
                    results,
                    top_k=rerank_top_k,
                )
                if reranked:
                    reranked_norm = [_normalize_result(r) for r in reranked]
                    MIN_RERANK_SCORE_AUDIO = 0.40
                    MIN_RERANK_SCORE_VISUAL = 0.15

                    def _has_real_rerank(r: Any) -> bool:
                        strat = str(getattr(r, "strategy", None) or r.get("strategy") or "")
                        return "+rerank" in strat

                    def _is_visual(r: Any) -> bool:
                        return _chunk_type_of(r) == "visual"

                    if any(_has_real_rerank(r) for r in reranked_norm):
                        filtered = [
                            r for r in reranked_norm
                            if float(getattr(r, "rerank_score", None) or r.get("rerank_score") or 0)
                            >= (MIN_RERANK_SCORE_VISUAL if _is_visual(r) else MIN_RERANK_SCORE_AUDIO)
                        ]
                        results = filtered if filtered else reranked_norm[:1]
                    else:
                        results = reranked_norm
            except Exception as e:
                logger.warning("reranker failed, keeping order: %s", e)

        if target_timestamps:
            results = sorted(_filter_modality(results), key=_timestamp_sort_key)
            final_limit = max(4, min(len(results), top_k or 6))
            return results[:final_limit]

        if visual_query:
            visuals = [item for item in results if _chunk_type_of(item) == "visual"]
            non_visuals = [item for item in results if _chunk_type_of(item) != "visual"]
            if visuals:
                results = visuals + non_visuals

        # ── FUZZY FALLBACK for yes_no queries ─────────────────────────────
        # When the question is yes/no and asks whether something was mentioned,
        # check if the key entity tokens appear in any retrieved chunk text.
        # If not, run a phonetic/substring fallback scan over ALL stored chunks.
        # This catches STT transcription errors (e.g. "Geoffrey" → "Jeffington")
        # without requiring any hardcoded name lists.
        if query_intent == "yes_no" and video_id and results:
            import re as _re2
            # Extract candidate entity tokens: capitalized words or words after
            # "mention", "say", "talk about", "discuss" etc.
            entity_tokens: List[str] = []
            # Named entity heuristic: capitalized words (length >= 4) in the original query
            for word in _re2.findall(r"\b[A-Z][a-z]{3,}\b", query):
                if word.lower() not in {
                    "does", "did", "the", "speaker", "mention", "talk", "about",
                    "this", "that", "video", "what", "when", "where", "who", "how",
                }:
                    entity_tokens.append(word)

            if entity_tokens:
                # Check if any token appears in any retrieved chunk text
                all_retrieved_text = " ".join(
                    (getattr(r, "text", None) or (r.get("text") if isinstance(r, dict) else None) or "").lower()
                    for r in results
                ).lower()
                missing = [t for t in entity_tokens if t.lower() not in all_retrieved_text]

                if missing and hasattr(store, "fuzzy_text_search"):
                    # Entity not found in retrieved chunks — run phonetic fallback
                    logger.info(
                        "yes_no fuzzy fallback: entity tokens %s not in retrieved chunks, scanning all chunks",
                        missing,
                    )
                    fuzzy_hits = store.fuzzy_text_search(
                        video_id=video_id,
                        tokens=missing,
                        top_k=5,
                    )
                    if fuzzy_hits:
                        # Merge fuzzy hits at the front (they're the most specific evidence)
                        seen_ids = {
                            getattr(r, "chunk_id", None) or (r.get("chunk_id") if isinstance(r, dict) else None)
                            for r in results
                        }
                        new_hits = [
                            r for r in fuzzy_hits
                            if (getattr(r, "chunk_id", None) or (r.get("chunk_id") if isinstance(r, dict) else None))
                            not in seen_ids
                        ]
                        if new_hits:
                            logger.info("yes_no fuzzy fallback: injecting %d new chunks", len(new_hits))
                            results = _dedupe(new_hits + list(results))
        # ──────────────────────────────────────────────────────────────────

        return results
    except Exception as e:
        logger.error("_do_search_sync FAILED: %s\n%s", e, traceback.format_exc())
        return []


# =====================================================================
#                         E N D P O I N T S
# =====================================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "version": "6.2.0",
        "components": {
            "transcriber":  transcriber is not None,
            "chunker":      chunker is not None,
            "embedder":     embedder is not None,
            "reranker":     reranker is not None,
            "vision":       vision is not None,
            "ocr_engine":   vision is not None and getattr(vision, "vlm_config", None) is not None and vision.vlm_config.ocr_fallback_enabled,
            "ocr_backend":  "per-video (easyocr/tesseract)",
            "store":        store is not None,
            "llm_client":   llm_client is not None,
            "agentic":      agentic is not None,
            "vqa_engine":   vqa_engine is not None,
            "frame_index":  _frame_index is not None,
            "metadata_registry": _metadata_registry is not None,
            "query_cache":  _query_cache is not None,
        },
    }


# =====================================================================
# AUTH ENDPOINTS
# =====================================================================

def _token_from_request(request: Request) -> Optional[str]:
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return request.headers.get("X-Session-Token", "").strip() or None


def _require_user(request: Request) -> dict:
    if _auth_db is None:
        raise HTTPException(status_code=503, detail="Auth service not available")
    token = _token_from_request(request)
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    user = _auth_db.validate_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    return user


class LoginRequest(BaseModel):
    email: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    display_name: str
    email: str
    password: str


class UpdateProfileRequest(BaseModel):
    display_name: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    bio: Optional[str] = None


class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str


@app.post("/auth/login")
async def auth_login(req: LoginRequest):
    if _auth_db is None:
        raise HTTPException(status_code=503, detail="Auth service not available")
    result = _auth_db.login(req.email, req.password)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    user, token = result
    return {"token": token, "user": _public_user_dict(user)}


@app.post("/auth/register")
async def auth_register(req: RegisterRequest):
    if _auth_db is None:
        raise HTTPException(status_code=503, detail="Auth service not available")
    if _auth_db.get_by_email(req.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    if _auth_db.get_by_username(req.username):
        raise HTTPException(status_code=409, detail="Username already taken")
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    try:
        user = _auth_db.create_user(
            username=req.username,
            display_name=req.display_name,
            email=req.email,
            password=req.password,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    token = _auth_db.create_session(user["id"])
    return {"token": token, "user": _public_user_dict(user)}


@app.post("/auth/logout")
async def auth_logout(request: Request):
    token = _token_from_request(request)
    if token and _auth_db is not None:
        _auth_db.delete_session(token)
    return {"status": "ok"}


@app.get("/auth/me")
async def auth_me(request: Request):
    user = _require_user(request)
    return _public_user_dict(user)


@app.put("/auth/me")
async def auth_update_me(req: UpdateProfileRequest, request: Request):
    user = _require_user(request)
    if req.email and req.email != user["email"]:
        existing = _auth_db.get_by_email(req.email)
        if existing and existing["id"] != user["id"]:
            raise HTTPException(status_code=409, detail="Email already in use")
    if req.username and req.username.lower() != user["username"]:
        existing = _auth_db.get_by_username(req.username)
        if existing and existing["id"] != user["id"]:
            raise HTTPException(status_code=409, detail="Username already taken")
    updated = _auth_db.update_profile(
        user["id"],
        display_name=req.display_name,
        username=req.username,
        email=req.email,
        bio=req.bio,
    )
    return _public_user_dict(updated)


@app.put("/auth/me/password")
async def auth_change_password(req: ChangePasswordRequest, request: Request):
    user = _require_user(request)
    if len(req.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    ok = _auth_db.change_password(user["id"], req.current_password, req.new_password)
    if not ok:
        raise HTTPException(status_code=401, detail="Current password is incorrect")
    return {"status": "ok"}


@app.post("/auth/me/avatar")
async def auth_upload_avatar(request: Request, file: UploadFile = File(...)):
    user = _require_user(request)
    content_type = file.content_type or "image/jpeg"
    if not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    data = await file.read()
    if len(data) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Avatar must be under 5 MB")
    filename = _auth_db.save_avatar(user["id"], data, content_type)
    return {"avatar_url": f"/avatars/{filename}"}


@app.get("/avatars/{filename}")
async def serve_avatar(filename: str):
    if _auth_db is None:
        raise HTTPException(status_code=503, detail="Auth service not available")
    # Sanitize: no path traversal
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = _auth_db.get_avatar_path(filename)
    if path is None:
        raise HTTPException(status_code=404, detail="Avatar not found")
    media_type = "image/png" if filename.endswith(".png") else "image/jpeg"
    return FileResponse(str(path), media_type=media_type)


@app.get("/auth/stats")
async def auth_stats(request: Request):
    _require_user(request)
    if _auth_db is None:
        return {}
    return _auth_db.stats()


@app.get("/cache/stats")
async def cache_stats_endpoint(request: Request):
    _require_user(request)
    if _query_cache is None:
        return {"enabled": False, "total_entries": 0}
    return _query_cache.stats()


@app.get("/debug/embedder")
async def debug_embedder():
    """Test endpoint to verify embedder is working."""
    if embedder is None:
        return {"error": "Embedder not initialized"}

    try:
        test_query = "test query"
        result = embedder.embed_query(test_query)

        if result is None:
            return {"error": "embed_query returned None"}

        # Vérifier si c'est un objet QueryEmbedding
        has_bge = hasattr(result, 'bge_dense')
        bge_dense = result.bge_dense if has_bge else None

        return {
            "status": "ok",
            "has_bge_dense": has_bge,
            "bge_dense_length": len(bge_dense) if bge_dense else 0,
            "bge_dense_sample": bge_dense[:5] if bge_dense else None,
            "result_type": str(type(result)),
            "result_attributes": [a for a in dir(result) if not a.startswith('_')][:20]
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}


@app.get("/debug/storage/{video_id}")
async def debug_storage(video_id: str):
    """Check if chunks for a video are actually in storage."""
    if store is None:
        return {"error": "Store not initialized"}

    try:
        # Vérifier les stats de la vidéo
        stats = store.get_stats(video_id=video_id)

        # Essayer de récupérer les chunks
        chunks = []
        for method in ["get_chunks", "list_chunks", "get_video_chunks"]:
            fn = getattr(store, method, None)
            if callable(fn):
                try:
                    chunks = fn(video_id=video_id) or []
                    break
                except Exception:
                    pass

        return {
            "video_id": video_id,
            "stats": stats,
            "chunks_found": len(chunks),
            "chunks_sample": [{
                "chunk_id": c.get("chunk_id") if isinstance(c, dict) else getattr(c, "chunk_id", None),
                "has_bge_dense": bool(c.get("bge_dense") if isinstance(c, dict) else getattr(c, "bge_dense", None))
            } for c in chunks[:3]] if chunks else []
        }
    except Exception as e:
        return {"error": str(e), "traceback": traceback.format_exc()}

@app.get("/")
async def serve_welcome():
    """Serves the marketing LANDING page (new design: project/VGIA.html).

    The functional dashboard lives at /app — NOT at the root. project/ is also
    mounted at the site root (see end of file) so the landing's relative assets
    (assets/*.png) and sibling pages (login.html, signup.html, pricing.html,
    documentation.html, whats-new.html) resolve correctly.
    """
    landing = PROJECT_DIR / "VGIA.html"
    if landing.exists():
        return FileResponse(path=str(landing), media_type="text/html")
    # Fallback to the previous welcome page if the new landing is missing.
    ui_file = BASE_DIR / "static" / "index.html"
    if ui_file.exists():
        return FileResponse(path=str(ui_file), media_type="text/html")
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content="<h1>askVGIA</h1><p>UI not built.</p>")


@app.get("/app")
@app.get("/app/")
async def serve_app():
    """Serves the main application dashboard (new frontend)."""
    app_file = BASE_DIR / "static" / "app" / "index.html"
    if app_file.exists():
        return FileResponse(path=str(app_file), media_type="text/html")
    # fallback: redirect to welcome
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")


@app.get("/login")
async def serve_login_redirect():
    # Auth now lives in the marketing site (project/login.html), served at
    # /login.html by the root mount. The old dashboard login was removed.
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/login.html")


@app.get("/app/{page:path}")
async def serve_app_page(page: str):
    """Serves inner HTML pages of the app dashboard."""
    # Only serve .html files through this route; assets are served by /app-assets mount
    if not page.endswith(".html"):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/app")
    app_file = BASE_DIR / "static" / "app" / page
    # Prevent path traversal
    try:
        app_file.resolve().relative_to((BASE_DIR / "static" / "app").resolve())
    except ValueError:
        raise HTTPException(status_code=403)
    if app_file.exists():
        return FileResponse(path=str(app_file), media_type="text/html")
    app_index = BASE_DIR / "static" / "app" / "index.html"
    if app_index.exists():
        return FileResponse(path=str(app_index), media_type="text/html")
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/")


@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: Optional[str] = Form(None),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    video_id = _generate_video_id(file.filename)
    if _video_is_already_indexed(video_id):
        logger.info("⏭️  /upload — %s already indexed", video_id)
        return JSONResponse(
            status_code=200,
            content={
                "video_id": video_id,
                "status": "already_indexed",
                "message": "This video is already indexed. Use /reindex to force.",
                "has_file": _find_video_file(video_id) is not None,
            },
        )
    suffix = Path(file.filename).suffix or ".mp4"
    dest = DATA_DIR / f"{video_id}{suffix}"
    try:
        with dest.open("wb") as buf:
            shutil.copyfileobj(file.file, buf)
        logger.info("📁 saved → %s", dest)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File save failed: {e}")
    background_tasks.add_task(
        _submit_ingest_to_executor, str(dest), video_id, language,
    )
    return JSONResponse(
        status_code=202,
        content={
            "video_id": video_id,
            "status": "processing",
            "message": "Ingestion started in background.",
            "filename": file.filename,
        },
    )


def _submit_ingest_to_executor(
    video_path: str, video_id: str, language: Optional[str],
) -> None:
    def _done_callback(fut):
        try:
            result = fut.result()
            logger.info("ingest done: %s — %s", video_id, result.get("status"))
        except Exception as e:
            logger.error("ingest error: %s — %s", video_id, e)

    future = _ingest_executor.submit(_run_ingest_sync, video_path, video_id, language)
    future.add_done_callback(_done_callback)


@app.post("/ingest")
async def ingest_video(req: IngestRequest, background_tasks: BackgroundTasks):
    video_path = req.video_path
    if not Path(video_path).is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {video_path}")
    video_id = req.video_id or _generate_video_id(Path(video_path).name)
    if _video_is_already_indexed(video_id):
        logger.info("⏭️  /ingest — %s already indexed", video_id)
        return JSONResponse(
            status_code=200,
            content={
                "video_id": video_id,
                "status": "already_indexed",
                "message": "Already indexed. Use /reindex to force.",
            },
        )
    src = Path(video_path)
    dest = DATA_DIR / f"{video_id}{src.suffix}"
    if not dest.exists():
        try:
            shutil.copy2(src, dest)
        except Exception:
            dest = src
    background_tasks.add_task(
        _submit_ingest_to_executor, str(dest), video_id, req.language,
    )
    return JSONResponse(
        status_code=202,
        content={
            "video_id": video_id,
            "status": "processing",
            "message": "Ingestion started in background.",
        },
    )


@app.post("/reindex")
async def reindex_video(req: IngestRequest, background_tasks: BackgroundTasks):
    video_path = req.video_path
    if not Path(video_path).is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {video_path}")
    video_id = req.video_id or _generate_video_id(Path(video_path).name)
    if store is not None:
        try:
            store.delete_video(video_id)
            logger.info("🗑️  deleted old index for %s", video_id)
        except Exception as e:
            logger.warning("delete_video failed: %s", e)
    if _frame_index is not None:
        try:
            _frame_index.delete_video(video_id) if hasattr(_frame_index, "delete_video") else 0
        except Exception as e:
            logger.warning("clear frames failed: %s", e)
    # Clear cached chapters so they are rebuilt fresh after re-indexing
    if _metadata_registry is not None:
        _rmeta = _metadata_registry.get(video_id)
        if _rmeta is not None and "chapters" in _rmeta.extra:
            del _rmeta.extra["chapters"]
            _rmeta.touch()
            _metadata_registry.register(_rmeta)
            logger.info("🗑️  cleared cached chapters for %s", video_id)
    src = Path(video_path)
    dest = DATA_DIR / f"{video_id}{src.suffix}"
    if not dest.exists():
        try:
            shutil.copy2(src, dest)
        except Exception:
            dest = src
    background_tasks.add_task(
        _submit_ingest_to_executor, str(dest), video_id, req.language,
    )
    return JSONResponse(
        status_code=202,
        content={
            "video_id": video_id,
            "status": "reindexing",
            "message": "Old chunks deleted. Re-ingestion started.",
        },
    )


@app.post("/download-yt")
async def download_youtube(
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    video_id: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
):
    try:
        import yt_dlp
    except ImportError:
        raise HTTPException(status_code=501, detail="yt-dlp not installed. Run: pip install yt-dlp")
    import uuid
    vid = (video_id or str(uuid.uuid4())[:8]).replace(" ", "_")
    out = DATA_DIR / f"{vid}.mp4"
    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": str(out),
        "quiet": True,
        "merge_output_format": "mp4",
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Download failed: {exc}")
    if _video_is_already_indexed(vid):
        return JSONResponse(
            status_code=200,
            content={"video_id": vid, "status": "already_indexed"},
        )
    background_tasks.add_task(_submit_ingest_to_executor, str(out), vid, language)
    return JSONResponse(
        status_code=202,
        content={"video_id": vid, "status": "processing", "message": "Download complete, ingestion started."},
    )


@app.get("/videos")
async def list_videos():
    if store is None:
        return JSONResponse(status_code=503, content={"videos": [], "count": 0, "error": "Store not available"})
    try:
        all_ids: List[str] = store.list_videos()
    except Exception as e:
        logger.error("list_videos failed: %s", e)
        return JSONResponse(status_code=500, content={"videos": [], "count": 0, "error": str(e)})
    videos = []
    for vid in all_ids:
        vid_str = str(vid) if not isinstance(vid, str) else vid
        video_file = _find_video_file(vid_str)
        stats: Dict[str, Any] = {}
        try:
            stats = store.get_stats(video_id=vid_str) or {}
        except Exception:
            pass
        total_chunks = int(stats.get("total_chunks") or stats.get("chunks") or 0)

        # BUG-4: enrich with video-level metadata
        meta: Optional[VideoMetadata] = None
        if _metadata_registry is not None:
            meta = _metadata_registry.get(vid_str)

        # Resolve thumbnail: look up the first visual chunk for this video
        thumbnail_url = None
        try:
            if _frame_index is not None and hasattr(_frame_index, "list_frames"):
                first_frames = _frame_index.list_frames(video_id=vid_str, limit=1)
                if first_frames:
                    thumb_chunk_id = first_frames[0].get("chunk_id", "")
                    if thumb_chunk_id:
                        thumbnail_url = _public_frame_url(thumb_chunk_id)
        except Exception:
            pass

        videos.append({
            "video_id": vid_str,
            "filename": video_file.name if video_file else None,
            # BUG-4: use LLM-generated title if available, fallback to filename
            "title": (meta.title if meta and meta.title else
                      (video_file.stem if video_file else vid_str)),
            "has_file": video_file is not None,
            "total_chunks": total_chunks,
            "file_url": f"/video/{vid_str}/stream" if video_file else None,
            "thumbnail_url": thumbnail_url,
            "languages": stats.get("languages", []),
            "speakers": stats.get("speakers", []),
            "filter_tags": stats.get("filter_tags", []),
            # BUG-4: new metadata fields
            "summary": meta.summary if meta else "",
            "keywords": meta.keywords if meta else [],
            "topics": meta.topics if meta else [],
            "language": meta.language if meta else stats.get("language", "unknown"),
            "has_visual": meta.has_visual if meta else False,
            "duration_seconds": meta.duration_seconds if meta else None,
        })
    return {"videos": videos, "count": len(videos)}


@app.get("/video/{video_id}")
async def video_info(video_id: str):
    video_file = _find_video_file(video_id)
    stats: Dict[str, Any] = {}
    if store is not None:
        try:
            stats = store.get_stats(video_id=video_id) or {}
        except Exception:
            pass
    # BUG-4: enrich with video-level metadata
    meta: Optional[VideoMetadata] = None
    if _metadata_registry is not None:
        meta = _metadata_registry.get(video_id)

    return {
        "video_id": video_id,
        "filename": video_file.name if video_file else None,
        "title": (meta.title if meta and meta.title else
                  (video_file.stem if video_file else video_id)),
        "has_file": video_file is not None,
        "total_chunks": int(stats.get("total_chunks", 0) or 0),
        "languages": stats.get("languages", []),
        "speakers": stats.get("speakers", []),
        "filter_tags": stats.get("filter_tags", []),
        "file_url": f"/video/{video_id}/stream" if video_file else None,
        # BUG-4: metadata fields
        "summary": meta.summary if meta else "",
        "keywords": meta.keywords if meta else [],
        "topics": meta.topics if meta else [],
        "language": meta.language if meta else "unknown",
        "has_visual": meta.has_visual if meta else False,
        "duration_seconds": meta.duration_seconds if meta else None,
        # CHAPTER-FIX: embed cached chapters so frontend never needs a separate slow call
        "chapters": (meta.extra.get("chapters") if meta and isinstance(meta.extra, dict) else None),
    }


@app.get("/video/{video_id}/stream")
async def stream_video(video_id: str, request: Request):
    video_file = _find_video_file(video_id)
    if video_file is None or not video_file.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    media_type, _ = mimetypes.guess_type(str(video_file))
    media_type = media_type or "video/mp4"
    file_size = video_file.stat().st_size

    range_header = request.headers.get("range")
    if not range_header:
        # No Range header — serve the full file with Accept-Ranges so the browser
        # knows it can seek without reconnecting.
        return StreamingResponse(
            _iter_file(video_file, 0, file_size - 1),
            status_code=200,
            media_type=media_type,
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Content-Disposition": f'inline; filename="{video_file.name}"',
            },
        )

    # Parse "bytes=start-end"
    try:
        range_val = range_header.replace("bytes=", "")
        parts = range_val.split("-")
        start = int(parts[0]) if parts[0] else 0
        end = int(parts[1]) if len(parts) > 1 and parts[1] else file_size - 1
    except Exception:
        raise HTTPException(status_code=416, detail="Invalid Range header")

    if start > end or start >= file_size:
        raise HTTPException(
            status_code=416,
            detail="Range Not Satisfiable",
            headers={"Content-Range": f"bytes */{file_size}"},
        )
    end = min(end, file_size - 1)
    chunk_size = end - start + 1

    return StreamingResponse(
        _iter_file(video_file, start, end),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(chunk_size),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{video_file.name}"',
        },
    )


def _iter_file(path: Path, start: int, end: int, chunk: int = 1 << 20):
    """Yield chunks of a file between byte positions start and end (inclusive)."""
    with path.open("rb") as f:
        f.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            data = f.read(min(chunk, remaining))
            if not data:
                break
            remaining -= len(data)
            yield data


@app.post("/search")
async def search(req: SearchRequest):
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Embedder or store not available")
    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(
        _search_executor,
        _do_search_sync,
        req.query, req.video_id, req.top_k, req.with_vision,
    )
    return {"query": req.query, "video_id": req.video_id, "results": results, "count": len(results)}


# =====================================================================
# BUG-7 FIX: /ask endpoint with man-in-the-loop clarification
# =====================================================================

@app.post("/ask")
async def ask(req: AskRequest):
    """
    v7.0: Direct answer pipeline — no clarification gate.
    Injects video metadata (title, summary, domain, topics) into every
    LLM call so the model is fully grounded in the specific video.

    CACHE: Before running the full pipeline we check the semantic query
    cache.  If a suffisamment similar question about the same video was
    asked before we return the cached answer immediately.  New answers
    are always saved to the cache for future reuse.
    """
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Embedder or store not available")

    loop = asyncio.get_running_loop()
    effective_question = req.reformulated_question or req.question
    llm_question = _expand_query_aliases(effective_question)

    if _query_cache is not None and req.use_cache:
        try:
            cached = _query_cache.lookup(
                question=effective_question,
                video_id=req.video_id,
            )
            if cached is not None and not _is_stale_cache_answer(effective_question, cached):
                # LANG-FIX: re-detect language from the CURRENT query, not the cached one.
                from core.llm_client import _detect_language_hint as _detect_lang_cache
                _current_lang = req.language_hint or _detect_lang_cache(effective_question)
                _cached_lang = (cached.get("llm") or {}).get("language_hint", "en") or "en"
                if _cached_lang != _current_lang:
                    # Language mismatch — skip cache and generate a fresh answer.
                    logger.info(
                        "QueryCache ask LANG-SKIP | cached_lang=%s current_lang=%s | q=%r",
                        _cached_lang, _current_lang, effective_question[:60],
                    )
                else:
                    cached["question"] = effective_question
                    cached["original_question"] = req.question
                    cached["video_id"] = req.video_id
                    cached["confirmed"] = True
                    if isinstance(cached.get("llm"), dict):
                        cached["llm"]["language_hint"] = _current_lang
                    return cached
        except Exception as _cache_err:
            logger.warning("QueryCache.lookup failed (ignored): %s", _cache_err)

    from core.llm_client import _detect_intent as _detect_query_intent
    from core.llm_client import classify_query_modality as _classify_modality_ask

    _intent_for_vision = _detect_query_intent(effective_question)
    # VISUAL-FIX: _detect_intent only returns "visual" for appearance/clothing queries.
    # Yes/no and factual queries about what is *shown* on screen (flags, logos, slides)
    # are classified as "yes_no" or "factual" by intent but STILL need visual chunks.
    # classify_query_modality is the authoritative modality signal — use it here too.
    _modality_for_ask = _classify_modality_ask(effective_question, use_llm=False)  # STEP-1: keyword rules, no LLM call

    # ── LVBENCH FIX ──────────────────────────────────────────────────
    # counting/event/key_info/temporal_grounding are visual-dominant and need
    # more frames retrieved. Bump top_k and force vision on for these intents.
    # Strip MCQ (A)/(B)/(C)/(D) options so the embedding query isn't polluted
    # by distractors. The FULL question (with options) still goes to the LLM.
    try:
        from core.llm_client import (
            detect_lvbench_intent as _detect_lv_real,
            lvbench_top_k as _lv_top_k,
            strip_mcq_options as _strip_mcq,
            _LV_VISUAL_INTENTS as _LV_VIS,
        )
    except Exception:
        _detect_lv_real = lambda q: None      # noqa: E731
        _lv_top_k = lambda i, k: (k or 8)     # noqa: E731
        _strip_mcq = lambda q: q              # noqa: E731
        _LV_VIS = set()

    # STEP-1: LVBench intent routing is DISABLED. It misrouted transcript /
    # educational questions to visual-only mode (e.g. "what happened when ...").
    # We keep strip_mcq (useful) but neutralize the intent so it no longer biases
    # modality, top_k, or graph mode. Re-enable by restoring _detect_lv_real.
    _LVBENCH_ROUTING_ENABLED = False
    _detect_lv = _detect_lv_real if _LVBENCH_ROUTING_ENABLED else (lambda q: None)

    _lv_intent = _detect_lv(effective_question)
    _search_question = _strip_mcq(effective_question)   # clean query for retrieval
    _effective_top_k = _lv_top_k(_lv_intent, req.top_k) if _lv_intent else req.top_k

    _effective_vision = (
        req.vision
        or (_intent_for_vision == "visual")
        or (_modality_for_ask in ("visual", "both"))
        or (_lv_intent in _LV_VIS)
    )

    # LANG-FIX: auto-detect language from query when frontend doesn't send language_hint.
    # This ensures Arabic/French questions always get answers in the right language.
    from core.llm_client import _detect_language_hint as _detect_lang
    _resolved_lang = req.language_hint or _detect_lang(effective_question)

    vector_future = loop.run_in_executor(
        _search_executor,
        _do_search_sync,
        _search_question,          # LVBENCH: MCQ-stripped query for retrieval
        req.video_id,
        _effective_top_k,          # LVBENCH: intent-aware top_k
        _effective_vision,
    )

    graph_future = None
    # LVBENCH: reasoning questions benefit from global GraphRAG (causal chains).
    _use_graph = req.use_graph or (_lv_intent == "reasoning")
    _graph_mode_eff = "global" if _lv_intent == "reasoning" else getattr(req, "graph_mode", "auto")
    if _use_graph and graph_search is not None and req.video_id:
        graph_future = loop.run_in_executor(
            _search_executor,
            lambda: graph_search.ask(
                question=effective_question,
                mode=_graph_mode_eff,
                video_id=req.video_id,
                top_k=_effective_top_k,
            ),
        )

    search_results: List[DotDict] = await vector_future

    graph_context_text: Optional[str] = None
    graph_answer_text: str = ""
    graph_answer_intent: str = "unknown"
    if graph_future is not None:
        try:
            _graph_result = await asyncio.wait_for(asyncio.shield(graph_future), timeout=25.0)
            if _graph_result:
                _graph_dict = _graph_result.get("graph") or {}
                graph_context_text = (
                    _graph_dict.get("context_text")
                    or _graph_result.get("graph_context")
                    or _graph_result.get("context_text")
                    or ""
                )
                graph_answer_text = str(_graph_result.get("answer") or "").strip()
                graph_answer_intent = str(_graph_result.get("intent") or "unknown")
                graph_vector_hits = _graph_result.get("vector") or _graph_result.get("results") or []
                if graph_vector_hits:
                    existing_ids = {
                        getattr(r, "chunk_id", None) or (r.get("chunk_id") if isinstance(r, dict) else None)
                        for r in search_results
                    }
                    for hit in graph_vector_hits:
                        cid = getattr(hit, "chunk_id", None) or (hit.get("chunk_id") if isinstance(hit, dict) else None)
                        if cid not in existing_ids:
                            search_results.append(hit)
                            existing_ids.add(cid)
        except asyncio.TimeoutError:
            logger.warning("Graph search exceeded 25s timeout in /ask — proceeding without graph context")
        except Exception as _graph_err:
            logger.warning("Graph enrichment in /ask failed (ignored): %s", _graph_err)

    video_meta = _get_video_metadata_dict(req.video_id)
    llm_answer: Optional[Dict[str, Any]] = None

    # ── VISION-FIRST ROUTING ──────────────────────────────────────────
    # For purely-visual questions ("is there an owl on screen", "what is on
    # the slide"), the text-RAG path answers from captions+transcript and
    # wrongly says "no" when the caption didn't happen to mention the object.
    # The VQA engine actually LOOKS at the retrieved frames (and has the
    # caption-keyword + object fallback), so route visual questions there
    # first. Fall back to text-RAG only if VQA finds no frames / low conf.
    if (
        req.llm
        and vqa_engine is not None
        and req.video_id
        and _modality_for_ask == "visual"
        and _detect_query_intent(effective_question) != "timestamp_lookup"
    ):
        try:
            _vqa = await loop.run_in_executor(
                _search_executor,
                lambda: vqa_engine.answer(
                    effective_question,
                    video_id=req.video_id,
                    top_n=req.top_k,
                    include_retrieval=True,
                ),
            )
            _vqa_ans = (_vqa or {}).get("answer", "") or ""
            _vqa_conf = float((_vqa or {}).get("confidence", 0.0) or 0.0)
            _no_frames = (
                "no relevant visual frames" in _vqa_ans.lower()
                or "no visual content indexed" in _vqa_ans.lower()
            )
            # Accept the VQA answer when it actually saw frames and isn't a
            # bare "I can't tell" with near-zero confidence.
            if _vqa_ans and not _no_frames and _vqa_conf >= 0.15:
                _frames_used = (_vqa or {}).get("frames_used", []) or []
                vqa_citations = [
                    {
                        "video_id": req.video_id,
                        "chunk_type": "visual",
                        "start_time": f.get("timestamp"),
                        "end_time": f.get("timestamp"),
                        "frame_path": f.get("path"),
                        "timestamp_source": "frame_exact",
                    }
                    for f in _frames_used
                ]
                llm_answer = {
                    "answer": _vqa_ans,
                    "model": (_vqa or {}).get("model", "vqa_engine"),
                    "used_fallback": False,
                    "intent": "visual",
                    "language_hint": _resolved_lang,
                    "citations": vqa_citations,
                    "confidence": _vqa_conf,
                    "mode": "vision_first",
                }
        except Exception as _vqa_err:
            logger.warning("VQA-first routing failed (falling back to text-RAG): %s", _vqa_err)

    if llm_answer is None and req.llm and (search_results or graph_context_text or graph_answer_text):
        if graph_answer_text and not search_results and not graph_context_text:
            llm_answer = {
                "answer": graph_answer_text,
                "model": "graph_search_fast_path",
                "used_fallback": False,
                "intent": graph_answer_intent,
                "language_hint": req.language_hint or "en",
                "citations": [],
                "confidence": 1.0,
            }

        if llm_answer is None and agentic is not None and search_results and not graph_context_text:
            try:
                llm_answer = await loop.run_in_executor(
                    _search_executor,
                    _call_agentic_search,
                    llm_question,
                    search_results,
                    req.video_id,
                    _resolved_lang,
                )
            except Exception as e:
                logger.warning("AgenticSearch failed, fallback to llm_client: %s", e)
                llm_answer = None

        if llm_answer is None and llm_client is not None and (search_results or graph_context_text):
            try:
                # VISUAL-FIX: pass modality so synthesize_answer can build
                # visual_context and use the right LLM system-prompt framing.
                _ask_modality = _modality_for_ask
                llm_answer = await loop.run_in_executor(
                    _search_executor,
                    lambda: llm_client.answer_question(
                        question=llm_question,
                        search_results=search_results,
                        preferred_model=None,
                        video_id=req.video_id,
                        language_hint=_resolved_lang,
                        video_metadata=video_meta,
                        graph_context=graph_context_text or "",
                        modality=_ask_modality,
                    ),
                )
            except Exception as e:
                logger.error("llm_client failed: %s", e)
                llm_answer = {
                    "answer": f"LLM error: {e}",
                    "model": None,
                    "used_fallback": True,
                    "intent": "unknown",
                    "language_hint": _resolved_lang,
                    "citations": [],
                }

        if llm_answer is None:
            llm_answer = {
                "answer": "LLM client is not available. Showing raw results.",
                "model": None,
                "used_fallback": True,
                "intent": "unknown",
                "language_hint": _resolved_lang,
                "citations": [],
            }

    answer_text = None
    if isinstance(llm_answer, dict):
        answer_text = llm_answer.get("answer")
    elif llm_answer is not None and hasattr(llm_answer, "answer"):
        answer_text = llm_answer.answer

    final_response = {
        "answer": answer_text,
        "llm": llm_answer,
        "results": search_results,
        "count": len(search_results),
        "question": effective_question,
        "original_question": req.question,
        "video_id": req.video_id,
        "confirmed": True,
        "video_metadata": video_meta,
        "graph_context": graph_context_text or "",
    }

    if (
        _query_cache is not None
        and req.use_cache
        and answer_text
        and not (isinstance(llm_answer, dict) and llm_answer.get("used_fallback"))
    ):
        try:
            _query_cache.save(
                question=effective_question,
                video_id=req.video_id,
                response=final_response,
            )
        except Exception as _cache_save_err:
            logger.warning("QueryCache.save failed (ignored): %s", _cache_save_err)

    return final_response


@app.post("/ask/stream")
async def ask_stream(req: AskRequest):
    """
    v7.0: Streaming answer with full video-aware context.
    Always streams directly — no clarification gate.
    Emits SSE 'meta' event first, then token-by-token streaming.

    CACHE: Checks the semantic query cache before running the full pipeline.
    On a hit, streams the cached answer token-by-token with a cache badge.
    On a miss, runs normally and saves the result for future reuse.
    """
    if store is None or embedder is None:
        raise HTTPException(status_code=503, detail="Embedder or store not available")

    effective_question = req.reformulated_question or req.question
    llm_question = _expand_query_aliases(effective_question)

    # LANG-FIX: resolve language BEFORE cache lookup so _resolved_lang is
    # available inside the cache try/except block (was causing UnboundLocalError).
    from core.llm_client import _detect_language_hint as _detect_lang_s
    _resolved_lang = req.language_hint or _detect_lang_s(effective_question)

    # ── CACHE LOOKUP ──────────────────────────────────────────────────
    if _query_cache is not None and req.use_cache:
        try:
            cached = _query_cache.lookup(
                question=effective_question,
                video_id=req.video_id,
            )
            if cached is not None and not _is_stale_cache_answer(effective_question, cached):
                # LANG-FIX: don't serve a cache entry whose answer is in a different
                # language than the current query. Let it fall through to a fresh LLM call.
                _cached_lang = (cached.get("llm") or {}).get("language_hint", "en") or "en"
                if _cached_lang != _resolved_lang:
                    logger.info(
                        "QueryCache stream LANG-SKIP | cached_lang=%s current_lang=%s | q=%r",
                        _cached_lang, _resolved_lang, effective_question[:60],
                    )
                    cached = None  # force a fresh generation

            if cached is not None and not _is_stale_cache_answer(effective_question, cached):
                logger.info(
                    "QueryCache stream HIT ✓ | video=%s | q=%r | score=%.3f",
                    req.video_id, effective_question[:60],
                    cached.get("_cache_similarity", 0),
                )

                def _stream_from_cache():
                    # ── meta preamble ──
                    llm_data = cached.get("llm") or {}
                    intent   = llm_data.get("intent", "")
                    # LANG-FIX: use the CURRENT query's language, not the cached one.
                    # The cached answer may have been generated for a query in a different language.
                    lang     = _resolved_lang
                    title    = (cached.get("video_metadata") or {}).get("title", "")
                    yield (
                        "event: meta\n"
                        "data: " + json.dumps({
                            "title":   title,
                            "intent":  intent,
                            "language": lang,
                            "cache_hit": True,
                            "cache_similarity": cached.get("_cache_similarity", 1.0),
                            "cache_method": cached.get("_cache_method", "cosine"),
                        }) + "\n\n"
                    )

                    # ── stream cached answer token-by-token ──
                    answer_text: str = (
                        cached.get("answer")
                        or llm_data.get("answer")
                        or ""
                    )
                    # Send in small chunks so the UI renders progressively
                    chunk_size = 8
                    for i in range(0, max(len(answer_text), 1), chunk_size):
                        token = answer_text[i : i + chunk_size]
                        if token:
                            yield f"data: {json.dumps({'token': token, 'done': False})}\n\n"

                    # ── done event with citations ──
                    citations = llm_data.get("citations", [])
                    model     = llm_data.get("model", "")
                    yield (
                        "data: " + json.dumps({
                            "token": "",
                            "done":  True,
                            "model": model,
                            "intent": intent,
                            "language": lang,
                            "citations": citations,
                            "used_fallback": False,
                            "cache_hit": True,
                            "cache_similarity": cached.get("_cache_similarity", 1.0),
                            "cache_hit_count":  cached.get("_cache_hit_count", 1),
                            "cache_method": cached.get("_cache_method", "cosine"),
                        }) + "\n\n"
                    )

                    # ── emit results event for audio+visual chunks ──
                    cached_results = cached.get("results") or []
                    if cached_results:
                        yield (
                            "event: results\n"
                            "data: " + json.dumps(cached_results) + "\n\n"
                        )

                return StreamingResponse(
                    _stream_from_cache(),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )
        except Exception as _cache_err:
            logger.warning("QueryCache.lookup (stream) failed (ignored): %s", _cache_err)
    # ──────────────────────────────────────────────────────────────────

    loop = asyncio.get_running_loop()

    # ── PARALLEL RETRIEVAL ────────────────────────────────────────────
    # Fire vector search AND graph search concurrently so neither blocks
    # the other.  Graph search previously ran *after* vector search,
    # adding its full latency (~81 s in the logs) to the critical path.
    _run = loop.run_in_executor  # shorthand

    # Auto-enable vision search for appearance/clothing/color questions
    from core.llm_client import _detect_intent as _di
    from core.llm_client import classify_query_modality as _classify_modality_stream
    # VISUAL-FIX: same fix as /ask — yes_no visual queries ("is there a flag?")
    # need visual chunks even though intent == "yes_no", not "visual".
    _modality_for_stream = _classify_modality_stream(effective_question, use_llm=False)  # STEP-1: keyword rules, no LLM call
    _effective_vision = req.vision or (_di(effective_question) == "visual") or (_modality_for_stream in ("visual", "both"))

    _vector_fut = _run(
        _search_executor,
        _do_search_sync,
        effective_question, req.video_id, req.top_k, _effective_vision,
    )

    _graph_fut = None
    if req.use_graph and graph_search is not None and req.video_id:
        _graph_mode = getattr(req, "graph_mode", "auto")
        _graph_fut = _run(
            _search_executor,
            lambda: graph_search.ask(
                question=effective_question,
                mode=_graph_mode,
                video_id=req.video_id,
                top_k=req.top_k,
            ),
        )

    # Always await vector search first — it must never be cancelled.
    # Then await graph search with a hard timeout so slow graph runs
    # don't block streaming. asyncio.wait_for would cancel both futures
    # on timeout, so we shield/wrap them independently.
    _GRAPH_TIMEOUT = 25.0
    _vector_result = await _vector_fut   # always complete, no timeout

    _graph_result = None
    if _graph_fut is not None:
        try:
            _graph_result = await asyncio.wait_for(
                asyncio.shield(_graph_fut),   # shield: timeout cancels wait_for wrapper, NOT _graph_fut itself
                timeout=_GRAPH_TIMEOUT,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Graph search exceeded %.0fs timeout — proceeding without graph context",
                _GRAPH_TIMEOUT,
            )
            _graph_result = None
        except Exception as _gex:
            logger.warning("Graph search failed: %s", _gex)
            _graph_result = None

    # Unpack vector results (may be an exception if the executor raised)
    if isinstance(_vector_result, Exception):
        logger.error("Vector search raised in parallel gather: %s", _vector_result)
        search_results: List[DotDict] = []
    else:
        search_results: List[DotDict] = _vector_result or []

    # v7.0: fetch video metadata for context injection
    video_meta = _get_video_metadata_dict(req.video_id)

    # ── UNIFIED GRAPH ENRICHMENT ──────────────────────────────────────
    # Merge graph context that was fetched in parallel above.
    graph_context_text: Optional[str] = None
    graph_answer_text: str = ""
    graph_answer_intent: str = "unknown"
    if _graph_result is not None:
        if isinstance(_graph_result, Exception):
            logger.warning("Graph enrichment in /ask/stream failed (ignored): %s", _graph_result)
        else:
            try:
                _gres = _graph_result
                # graph_search.ask() returns nodes inside _gres["graph"]
                # (a GraphSearchResult.as_dict()), not at the top level.
                _graph_dict = _gres.get("graph") or {}
                graph_context_text = (
                    _graph_dict.get("context_text")      # normal path: GraphSearchResult.as_dict()
                    or _gres.get("graph_context")        # speaker-count fast-path key
                    or _gres.get("context_text")         # legacy fallback
                    or ""
                )
                graph_answer_text = str(_gres.get("answer") or "").strip()
                graph_answer_intent = str(_gres.get("intent") or "unknown")
                _graph_node_count = len(_graph_dict.get("nodes") or [])
                # Merge graph's vector hits into search_results (dedup by chunk_id)
                graph_vector_hits = _gres.get("vector") or _gres.get("results") or []
                if graph_vector_hits:
                    existing_ids = {getattr(r, "chunk_id", None) or r.get("chunk_id") for r in search_results}
                    for hit in graph_vector_hits:
                        cid = getattr(hit, "chunk_id", None) or (hit.get("chunk_id") if isinstance(hit, dict) else None)
                        if cid not in existing_ids:
                            search_results.append(hit)
                            existing_ids.add(cid)
                logger.info(
                    "unified graph context | graph_nodes=%d graph_context_len=%d merged_results=%d",
                    _graph_node_count, len(graph_context_text), len(search_results),
                )
            except Exception as _graph_err:
                logger.warning("Graph enrichment merge failed (ignored): %s", _graph_err)
    # ─────────────────────────────────────────────────────────────────

    def _generate():
        accumulated_tokens: List[str] = []
        saved_citations: List[Any]    = []
        saved_model:     str          = ""
        saved_intent:    str          = ""
        saved_used_fallback: bool     = False

        if graph_answer_text and not search_results and not graph_context_text:
            saved_model = "graph_search_fast_path"
            saved_intent = graph_answer_intent
            meta = {
                "model": saved_model,
                "used_fallback": False,
                "intent": saved_intent,
                "language_hint": _resolved_lang,
            }
            yield "event: meta\n" + "data: " + json.dumps(meta) + "\n\n"
            for token in re.findall(r".{1,8}", graph_answer_text):
                accumulated_tokens.append(token)
                yield "data: " + json.dumps({"token": token}) + "\n\n"
            yield "data: " + json.dumps({
                "done": True,
                "citations": [],
                "model": saved_model,
                "used_fallback": False,
                "intent": saved_intent,
            }) + "\n\n"
        else:
            # VISUAL-FIX: pass modality so stream_answer builds visual_context
            # and uses the right LLM framing for all visual question types.
            for sse_chunk in llm_client.stream_answer(
                llm_question,
                search_results,
                language_hint=_resolved_lang,
                video_metadata=video_meta,
                graph_context=graph_context_text,
                modality=_modality_for_stream,
                benchmark=(getattr(req, "benchmark", None) or DEFAULT_BENCHMARK),
            ):
                yield sse_chunk

                # ── Parse yielded SSE chunks to rebuild the full answer ──
                # Chunks are either "event: meta\ndata: {...}\n\n"
                # or plain "data: {...}\n\n"
                # We must handle the meta event specially to capture intent.
                chunk_stripped = sse_chunk.strip()
                if chunk_stripped.startswith("event: meta"):
                    # Extract the data line from the meta event
                    for line in chunk_stripped.split("\n"):
                        line = line.strip()
                        if line.startswith("data: "):
                            try:
                                meta_obj = json.loads(line[6:])
                                saved_intent = meta_obj.get("intent", "") or ""
                            except Exception:
                                pass
                    continue

                for line in sse_chunk.split("\n"):
                    line = line.strip()
                    if not line.startswith("data: "):
                        continue
                    try:
                        obj = json.loads(line[6:])
                    except Exception:
                        continue
                    token = obj.get("token", "")
                    if obj.get("done"):
                        saved_citations      = obj.get("citations", [])
                        saved_model          = obj.get("model", "")
                        saved_used_fallback  = bool(obj.get("used_fallback", False))
                        # intent may also arrive in the done event
                        if obj.get("intent"):
                            saved_intent = obj.get("intent", "")
                    elif token:
                        accumulated_tokens.append(token)

        # ── EMIT RESULTS EVENT so frontend gets all audio+visual chunks ──
        # The LLM citations only contain items used for generation (often
        # visual-heavy).  By emitting the full search_results here, the
        # frontend can render both audio (speech) and visual segments.
        try:
            serializable_results = [dict(r) for r in search_results]
            yield (
                "event: results\n"
                "data: " + json.dumps(serializable_results) + "\n\n"
            )
        except Exception as _res_err:
            logger.warning("Failed to emit results SSE event: %s", _res_err)
        # ──────────────────────────────────────────────────────────────

        # ── QA LOG: append this question + answer + grounding to per-video JSON
        try:
            from core.qa_logger import log_qa
            _qa_answer = "".join(accumulated_tokens).strip()
            if _qa_answer:
                log_qa(
                    QA_LOG_PATH,
                    req.video_id or "",
                    req.question,
                    _qa_answer,
                    search_results,
                    extra={"benchmark": (getattr(req, "benchmark", None) or DEFAULT_BENCHMARK)},
                )
        except Exception as _qa_err:
            logger.warning("QA logging failed (ignored): %s", _qa_err)
        # ──────────────────────────────────────────────────────────────

        # ── CACHE SAVE after streaming completes ──────────────────────
        if _query_cache is not None and req.use_cache and not saved_used_fallback:
            full_answer = "".join(accumulated_tokens)
            if full_answer.strip():
                try:
                    # Serialize search_results as plain dicts (DotDict is a
                    # dict subclass so this is a safe shallow copy)
                    serializable_results = [dict(r) for r in search_results]
                    cache_response = {
                        "answer":   full_answer,
                        "llm": {
                            "answer":        full_answer,
                            "model":         saved_model,
                            "used_fallback": False,
                            "intent":        saved_intent,
                            "language_hint": req.language_hint or "",
                            "citations":     saved_citations,
                        },
                        "results":          serializable_results,
                        "count":            len(serializable_results),
                        "question":         effective_question,
                        "original_question": req.question,
                        "video_id":         req.video_id,
                        "confirmed":        True,
                        "video_metadata":   video_meta,
                    }
                    _query_cache.save(
                        question=effective_question,
                        video_id=req.video_id,
                        response=cache_response,
                    )
                except Exception as _cache_save_err:
                    logger.warning(
                        "QueryCache.save (stream) failed (ignored): %s",
                        _cache_save_err,
                    )
        # ──────────────────────────────────────────────────────────────

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


def _call_agentic_search(
    question: str,
    results: List[DotDict],
    video_id: Optional[str],
    language_hint: Optional[str],
) -> Dict[str, Any]:
    if agentic is None:
        return None
    answer_obj = _call(
        agentic,
        ["answer", "ask", "synthesize", "run"],
        question=question,
        results=results,
        video_id=video_id,
        language_hint=language_hint,
    )
    if isinstance(answer_obj, dict):
        return answer_obj
    elif hasattr(answer_obj, "answer"):
        return {
            "answer": answer_obj.answer,
            "model": getattr(answer_obj, "model", None),
            "used_fallback": getattr(answer_obj, "used_fallback", False),
            "intent": getattr(answer_obj, "intent", "unknown"),
            "language_hint": getattr(answer_obj, "language_hint", "en"),
            "citations": getattr(answer_obj, "citations", []),
        }
    else:
        return {"answer": str(answer_obj), "model": None}


@app.delete("/video/{video_id}")
async def delete_video(video_id: str):
    deleted_store = False
    deleted_file = False
    if store is not None:
        try:
            store.delete_video(video_id)
            deleted_store = True
        except Exception as e:
            logger.warning("delete_video store: %s", e)
    deleted_graph = False
    if graph_store is not None:
        try:
            graph_store.delete_video(video_id)
            graph_store.save()
            deleted_graph = True
        except Exception as e:
            logger.warning("delete_video graph: %s", e)
    if _frame_index is not None:
        try:
            _frame_index.delete_video(video_id) if hasattr(_frame_index, "delete_video") else None
        except Exception as e:
            logger.warning("delete from frame index: %s", e)
    video_file = _find_video_file(video_id)
    if video_file and video_file.exists():
        try:
            video_file.unlink()
            deleted_file = True
        except Exception as e:
            logger.warning("delete file: %s", e)
    # BUG-4: remove metadata record too
    deleted_meta = False
    if _metadata_registry is not None:
        try:
            deleted_meta = _metadata_registry.delete(video_id)
        except Exception as e:
            logger.warning("delete metadata failed: %s", e)

    # CACHE: purge all cached queries for this video
    deleted_cache = 0
    if _query_cache is not None:
        try:
            deleted_cache = _query_cache.delete_video(video_id)
        except Exception as e:
            logger.warning("QueryCache.delete_video failed: %s", e)

    return {
        "video_id": video_id,
        "deleted_store": deleted_store,
        "deleted_file": deleted_file,
        "deleted_meta": deleted_meta,
        "deleted_graph": deleted_graph,
        "deleted_cache_entries": deleted_cache,
    }


@app.get("/stats")
async def global_stats():
    if store is None:
        return {"error": "Store not available"}
    try:
        return store.get_stats() or {}
    except Exception as e:
        return {"error": str(e)}


@app.get("/stats/{video_id}")
async def video_stats(video_id: str):
    if store is None:
        return {"error": "Store not available"}
    try:
        return store.get_stats(video_id=video_id) or {}
    except Exception as e:
        return {"error": str(e)}


@app.get("/ingestion/status/{video_id}")
async def ingestion_status(video_id: str):
    indexed = _video_is_already_indexed(video_id)
    video_file = _find_video_file(video_id)
    stats: Dict[str, Any] = {}
    if store is not None and indexed:
        try:
            stats = store.get_stats(video_id=video_id) or {}
        except Exception:
            pass
    return {
        "video_id": video_id,
        "indexed": indexed,
        "has_file": video_file is not None,
        "total_chunks": int(stats.get("total_chunks", 0) or 0),
        "status": "ready" if indexed else "not_indexed",
    }


@app.get("/ollama/models")
async def ollama_models():
    if llm_client is None:
        return {"error": "llm_client not available", "models": []}
    try:
        list_fn = getattr(llm_client, "_list_models", None)
        choose_fn = getattr(llm_client, "_choose_model", None)
        models = list_fn() if callable(list_fn) else []
        chosen = choose_fn() if callable(choose_fn) else None
        return {
            "models": [
                {"name": m.get("name"), "family": m.get("family"), "size": m.get("parameter_size")}
                for m in models
            ],
            "count": len(models),
            "chosen": chosen,
        }
    except Exception as e:
        return {"error": str(e), "models": []}


# ── GraphRAG models ──────────────────────────────────────────────────

class GraphAskRequest(BaseModel):
    question: str
    mode: str = "auto"
    video_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)
    language_hint: Optional[str] = None


class GraphLocalRequest(BaseModel):
    query: Optional[str] = None
    seed_entities: Optional[List[str]] = None
    hops: int = 1
    max_nodes: int = 25
    video_id: Optional[str] = None


class GraphGlobalRequest(BaseModel):
    query: str
    top_k: int = 3
    video_id: Optional[str] = None


@app.get("/graph/health")
async def graph_health():
    gs  = graph_store
    gx  = graph_extractor
    gb  = graph_builder
    gsr = graph_search
    ready = all(x is not None for x in (gs, gx, gb, gsr))
    info: Dict[str, Any] = {
        "ready": ready,
        "graphrag_available": GRAPHRAG_AVAILABLE,
        "graph_store":     gs  is not None,
        "graph_extractor": gx  is not None,
        "graph_builder":   gb  is not None,
        "graph_search":    gsr is not None,
        "components": {
            "graph_store":     gs  is not None,
            "graph_extractor": gx  is not None,
            "graph_builder":   gb  is not None,
            "graph_search":    gsr is not None,
        },
    }
    if gs is not None:
        try:
            stats_fn = getattr(gs, "stats", None) or getattr(gs, "get_stats", None)
            info["stats"] = stats_fn() if callable(stats_fn) else {
                "n_nodes": int(gs.graph.number_of_nodes()),
                "n_edges": int(gs.graph.number_of_edges()),
            }
        except Exception as e:
            info["stats_error"] = str(e)
    return info


@app.get("/graph/stats")
async def graph_stats():
    if graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not available")
    return graph_store.stats()


@app.get("/graph/video/{video_id}")
async def graph_video(video_id: str, max_nodes: int = 150):
    """Return the entity/relation subgraph scoped to a single video, for viz."""
    if graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not available")

    video_id = (video_id or "").strip()
    try:
        sub = graph_store.subgraph_for_video(video_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"subgraph_for_video failed: {e}")

    nodes = []
    for key, d in sub.nodes(data=True):
        nodes.append({
            "id": key,
            "name": d.get("name", key),
            "type": d.get("type", "OTHER"),
            "description": d.get("description", ""),
            "n_mentions": int(d.get("n_mentions", 0)),
        })
    # Cap to the most-mentioned entities so large graphs stay renderable
    nodes.sort(key=lambda n: -n["n_mentions"])
    nodes = nodes[:max_nodes]
    kept_ids = {n["id"] for n in nodes}

    edges = []
    for u, v, k, d in sub.edges(keys=True, data=True):
        if u not in kept_ids or v not in kept_ids:
            continue
        edges.append({
            "source": u,
            "target": v,
            "type": d.get("type", k),
            "description": d.get("description", ""),
            "weight": float(d.get("weight", 1.0)),
        })

    return {
        "video_id": video_id,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


@app.post("/graph/ask")
async def graph_ask(req: GraphAskRequest):
    if graph_search is None:
        raise HTTPException(status_code=503, detail="GraphSearch not available")

    # ── CACHE LOOKUP ──────────────────────────────────────────────────
    # Prefix with "graph:" so graph/ask and /ask don't share cache slots
    _cache_key = f"graph:{req.question}"
    if _query_cache is not None:
        try:
            cached = _query_cache.lookup(
                question=_cache_key,
                video_id=req.video_id,
            )
            if cached is not None:
                cached["_from_graph_cache"] = True
                logger.info("graph/ask CACHE HIT | video=%s | q=%r", req.video_id, req.question[:60])
                return cached
        except Exception as _ce:
            logger.warning("QueryCache.lookup (graph/ask) failed (ignored): %s", _ce)
    # ─────────────────────────────────────────────────────────────────

    loop = asyncio.get_running_loop()
    out = await loop.run_in_executor(
        _search_executor,
        lambda: graph_search.ask(
            question=req.question,
            mode=req.mode,
            video_id=req.video_id,
            top_k=req.top_k,
            language=req.language_hint,
        ),
    )

    # ── CACHE SAVE ────────────────────────────────────────────────────
    if _query_cache is not None and out and out.get("answer"):
        try:
            _query_cache.save(
                question=_cache_key,
                video_id=req.video_id,
                response=out,
            )
        except Exception as _ce:
            logger.warning("QueryCache.save (graph/ask) failed (ignored): %s", _ce)
    # ─────────────────────────────────────────────────────────────────

    return out


@app.post("/graph/local")
async def graph_local(req: GraphLocalRequest):
    if graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not available")
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        _search_executor,
        lambda: graph_store.local_search(
            query=req.query,
            seed_entities=req.seed_entities,
            hops=req.hops,
            max_nodes=req.max_nodes,
            video_id=req.video_id,
        ),
    )
    return res.as_dict()


@app.post("/graph/global")
async def graph_global(req: GraphGlobalRequest):
    if graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not available")
    loop = asyncio.get_running_loop()
    res = await loop.run_in_executor(
        _search_executor,
        lambda: graph_store.global_search(query=req.query, top_k=req.top_k, video_id=req.video_id),
    )
    return res.as_dict()


@app.get("/graph/entity/{entity_name}")
async def graph_entity(entity_name: str):
    if graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not available")
    key = graph_store._resolve_node(entity_name)
    if not key or key not in graph_store.graph:
        raise HTTPException(status_code=404, detail=f"Entity '{entity_name}' not found")
    d = graph_store.graph.nodes[key]
    return {
        "key": key,
        "name": d.get("name", key),
        "type": d.get("type", "OTHER"),
        "description": d.get("description", ""),
        "aliases": sorted(d.get("aliases", set())),
        "video_ids": sorted(d.get("video_ids", set())),
        "n_mentions": d.get("n_mentions", 0),
        "mentions": d.get("mentions", [])[:50],
        "neighbors_in": [
            {"src": graph_store.graph.nodes[u].get("name", u), "type": k}
            for u, _, k in graph_store.graph.in_edges(key, keys=True)
        ][:50],
        "neighbors_out": [
            {"dst": graph_store.graph.nodes[v].get("name", v), "type": k}
            for _, v, k in graph_store.graph.out_edges(key, keys=True)
        ][:50],
    }


@app.get("/graph/communities")
async def graph_communities(top_k: int = 20):
    if graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not available")
    comms = sorted(graph_store.communities.values(), key=lambda c: -c.size)[:top_k]
    return {"count": len(comms), "communities": [c.as_dict() for c in comms]}


@app.post("/graph/enrich/{video_id}")
async def graph_enrich_video(video_id: str):
    """
    (Re-)run enrich_from_video_store for an already-ingested video.
    Call this once after updating graph_store.py to inject speaker nodes
    into videos that were processed before the fix was applied.
    """
    if graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not available")
    if store is None:
        raise HTTPException(status_code=503, detail="VideoStore not available")
    # FIX-1: strip whitespace that may sneak in via URL encoding
    # (e.g. POST /graph/enrich/%205d92d66e_one has a leading space)
    video_id = video_id.strip()
    if not video_id:
        raise HTTPException(status_code=422, detail="video_id must not be empty")
    try:
        stats = graph_store.enrich_from_video_store(
            video_store=store, video_id=video_id,
        )
        graph_store.save()
        return {"video_id": video_id, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/graph/enrich_all")
async def graph_enrich_all():
    """Enrich speaker nodes for ALL videos already in the store."""
    if graph_store is None or store is None:
        raise HTTPException(status_code=503, detail="GraphStore or VideoStore not available")
    results = {}
    for vid in store.list_videos():
        try:
            stats = graph_store.enrich_from_video_store(
                video_store=store, video_id=vid,
            )
            results[vid] = stats
        except Exception as e:
            results[vid] = {"error": str(e)}
    try:
        graph_store.save()
    except Exception:
        pass
    return results


@app.post("/graph/rebuild_communities")
async def graph_rebuild_communities(summarize: bool = True):
    if graph_builder is None:
        raise HTTPException(status_code=503, detail="GraphBuilder not available")
    loop = asyncio.get_running_loop()
    n = await loop.run_in_executor(
        _ingest_executor,
        lambda: graph_builder.rebuild_communities(summarize=summarize, save=True),
    )
    return {"n_communities": n}


@app.delete("/graph/video/{video_id}")
async def graph_delete_video(video_id: str):
    if graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not available")
    counts = graph_store.delete_video(video_id)
    try:
        graph_store.save()
    except Exception:
        pass
    return {"video_id": video_id, **counts}


@app.get("/graph/export")
async def graph_export():
    if graph_store is None:
        raise HTTPException(status_code=503, detail="GraphStore not available")
    return graph_store.to_json_dict()


# ── Vision endpoints ─────────────────────────────────────────────────

def _public_frame_url(chunk_id: str) -> str:
    return f"/vision/frame/{chunk_id}"


def _decorate_visual_result(item: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(item or {})
    chunk_id = str(row.get("chunk_id", "") or "")
    row["chunk_type"] = row.get("chunk_type") or "visual"
    if chunk_id:
        row["thumbnail_url"] = _public_frame_url(chunk_id)
    return row


class VisionAskRequest(BaseModel):
    question: str
    video_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=12)
    include_retrieval: bool = True
    timestamp: Optional[float] = Field(default=None, ge=0, description="If given, find frame nearest this timestamp (seconds) instead of CLIP search")


class VisionBatchAskRequest(BaseModel):
    questions: List[str]
    video_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=12)
    include_retrieval: bool = True


class VisionSearchRequest(BaseModel):
    query: str
    video_id: Optional[str] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=50)


class KeyframeCaptionRequest(BaseModel):
    frame_path: str
    question: Optional[str] = None


@app.get("/vision/health")
def vision_health():
    if vision is None:
        return {"ok": False, "reason": "VisionIndexer not initialised"}
    frame_stats = _frame_index.get_stats() if _frame_index is not None and hasattr(_frame_index, "get_stats") else {}
    return {
        "ok": vision.vlm.is_available() if vision.vlm else False,
        "vlm_backend": vision.vlm.backend_name if vision.vlm else None,
        "vlm_model": vision.vlm.model_name if vision.vlm else None,
        "clip_backend": vision.clip.backend_name if vision.clip else None,
        "clip_dim": vision.clip.dim if vision.clip else 0,
        "ocr_fallback": vision._ocr is not None if hasattr(vision, '_ocr') else False,
        "ocr_engine_active": hasattr(vision, "vlm_config") and vision.vlm_config.ocr_fallback_enabled,
        "ocr_engine_backend": "per-video (easyocr → tesseract fallback)",
        "ocr_languages": "auto-detected from video language",
        "enabled": vision.enabled if hasattr(vision, 'enabled') else False,
        "vqa_ready": vqa_engine is not None,
        "visual_clip": frame_stats,
    }


@app.get("/vision/stats")
def vision_stats():
    if _frame_index is None or not hasattr(_frame_index, "get_stats"):
        return {
            "total_frames": 0,
            "videos": {},
            "vlm_backend": vision.vlm.backend_name if vision and vision.vlm else None,
        }
    stats = _frame_index.get_stats()
    stats["vlm_backend"] = vision.vlm.backend_name if vision and vision.vlm else None
    return stats


@app.post("/vision/ask")
def vision_ask(req: VisionAskRequest):
    if vqa_engine is None:
        raise HTTPException(503, "VQA engine not available (need CLIP + VLM).")
    try:
        # If a timestamp is given → use the timestamp index to find the exact frame.
        # This is the correct path for "what is on screen at 02:30?" questions.
        if req.timestamp is not None and req.video_id and vision is not None:
            result = vision.answer_at_timestamp(
                req.video_id,
                req.timestamp,
                req.question,
            )
            return {
                "question": req.question,
                "video_id": req.video_id,
                "answer": result.answer,
                "confidence": result.confidence,
                "timestamp": req.timestamp,
                "frames_used": [
                    {"path": p, "timestamp": t}
                    for p, t in (result.frames_used or [])
                ],
                "mode": "vision_timestamp",
            }

        # No timestamp → fall back to CLIP vector search across all frames.
        payload = vqa_engine.answer(
            req.question,
            video_id=req.video_id,
            top_n=req.top_k,
            include_retrieval=req.include_retrieval,
        )
        payload["frames"] = [_decorate_visual_result(r) for r in payload.get("frames", [])]
        return payload
    except Exception as e:
        logger.exception("/vision/ask failed")
        raise HTTPException(500, f"VQA failed: {e}")


@app.post("/vision/at")
def vision_at_timestamp(req: VisionAskRequest):
    """
    Answer a visual question about a specific timestamp.
    Uses the VideoTimestampIndex to find the nearest indexed frame —
    no CLIP vector search involved. Works even if CLIP is mock/disabled.

    Example: POST /vision/at
    { "video_id": "abc", "timestamp": 150.0, "question": "what is on screen?" }
    """
    if vision is None:
        raise HTTPException(503, "Vision indexer not available.")
    if req.timestamp is None:
        raise HTTPException(422, "timestamp (seconds) is required for this endpoint.")
    if not req.video_id:
        raise HTTPException(422, "video_id is required.")
    try:
        result = vision.answer_at_timestamp(
            req.video_id,
            req.timestamp,
            req.question or "Describe everything visible in this frame in detail.",
        )
        return {
            "question": req.question,
            "video_id": req.video_id,
            "answer": result.answer,
            "confidence": result.confidence,
            "timestamp": req.timestamp,
            "frames_used": [
                {"path": p, "timestamp": t}
                for p, t in (result.frames_used or [])
            ],
            "mode": "vision_at_timestamp",
        }
    except Exception as e:
        logger.exception("/vision/at failed")
        raise HTTPException(500, f"Timestamp VQA failed: {e}")


@app.post("/vision/batch-ask")
def vision_batch_ask(req: VisionBatchAskRequest):
    if vqa_engine is None:
        raise HTTPException(503, "VQA engine not available (need CLIP + VLM).")
    max_q = int(_cfg("vision_batch_vqa_max_questions", 8) or 8)
    if not req.questions:
        raise HTTPException(422, "questions must not be empty")
    if len(req.questions) > max_q:
        raise HTTPException(422, f"Too many questions: max={max_q}")
    try:
        payload = vqa_engine.batch_answer(req.questions, video_id=req.video_id, top_n=req.top_k, include_retrieval=req.include_retrieval)
        payload["frames"] = [_decorate_visual_result(r) for r in payload.get("frames", [])]
        return payload
    except Exception as e:
        logger.exception("/vision/batch-ask failed")
        raise HTTPException(500, f"batch VQA failed: {e}")


@app.post("/vision/search")
def vision_search(req: VisionSearchRequest):
    if vqa_engine is None:
        raise HTTPException(503, "Visual search unavailable (CLIP disabled).")
    results = vqa_engine.retrieve(req.query, top_k=req.top_k, video_id=req.video_id)
    results = [_decorate_visual_result(r) for r in results]
    return {
        "query": req.query,
        "video_id": req.video_id,
        "answer": f"{len(results)} frame(s) visuels trouvés.",
        "results": results,
        "mode": "vision_search",
    }


@app.post("/vision/caption")
def vision_caption(req: KeyframeCaptionRequest):
    if vision is None or vision.vlm is None:
        raise HTTPException(503, "VLM not available.")
    if not os.path.exists(req.frame_path):
        raise HTTPException(404, f"frame not found: {req.frame_path}")
    if req.question:
        r = vision.vlm.vqa(req.frame_path, req.question)
        return r.to_dict() if hasattr(r, 'to_dict') else {"answer": str(r)}
    return {
        "frame_path": req.frame_path,
        "caption": vision.vlm.caption(req.frame_path),
        "short_caption": vision.vlm.short_caption(req.frame_path),
        "tags": vision.vlm.tags(req.frame_path),
        "ocr": vision.vlm.ocr(req.frame_path),
        "model": vision.vlm.model_name,
        "backend": vision.vlm.backend_name,
    }


def _resolve_frame_path(frame_path: str) -> Optional[str]:
    """
    Resolve a stored frame_path to an existing file, trying multiple
    strategies to handle relative vs absolute paths and CWD drift.

    Also handles Windows absolute paths (e.g. C:\\Users\\...) when the
    server is running on the same Windows machine — Path() handles these
    natively on Windows, and we guard against cross-platform mismatch.

    Returns the resolved absolute path string, or None if not found.
    """
    if not frame_path:
        return None

    p = Path(frame_path)

    # 1. Try as-is (already absolute, or CWD matches)
    if p.exists():
        return str(p.resolve())

    # 2. Try relative to BASE_DIR (most common: server started from a
    #    parent directory but frames were saved relative to the project root)
    candidate = BASE_DIR / p
    if candidate.exists():
        return str(candidate.resolve())

    # 3. Try relative to CWD explicitly
    candidate_cwd = Path.cwd() / p
    if candidate_cwd.exists():
        return str(candidate_cwd.resolve())

    # 4. Try relative to each parent of BASE_DIR (one level up, e.g. monorepo)
    candidate2 = BASE_DIR.parent / p
    if candidate2.exists():
        return str(candidate2.resolve())

    # 5. If the stored path is an absolute path from another drive/user
    #    (e.g. stored as C:\Users\OtherUser\... but now running as different user),
    #    try extracting just the filename and scan for it under BASE_DIR/data/frames
    p_name = p.name
    if p_name:
        for ext_root in [BASE_DIR / "data" / "frames", Path("data") / "frames"]:
            if ext_root.exists():
                for match in ext_root.rglob(p_name):
                    return str(match.resolve())

    return None

def _scan_frames_dir_for_chunk(chunk_id: str) -> Optional[str]:
    """
    Last-resort fallback: walk the frames directory tree looking for any
    file whose name starts with the chunk_id.  Handles the case where the
    metadata record is present but frame_path is empty/stale.

    Searches multiple candidate roots so it works regardless of CWD.
    """
    cid_lower = chunk_id.lower()

    # All candidate roots to search, in priority order
    candidates: List[Path] = [
        BASE_DIR / "data" / "frames",
        Path("data") / "frames",
        Path.cwd() / "data" / "frames",
        BASE_DIR.parent / "data" / "frames",
    ]
    # Remove duplicates while preserving order
    seen: set = set()
    roots: List[Path] = []
    for c in candidates:
        try:
            resolved = c.resolve()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                roots.append(resolved)
        except Exception:
            pass

    for frames_root in roots:
        if not frames_root.exists():
            continue
        for ext in ("*.jpg", "*.jpeg", "*.png"):
            for img in frames_root.rglob(ext):
                stem = img.stem.lower()
                if stem.startswith(cid_lower) or cid_lower in stem:
                    return str(img)

    return None


def _get_frame_path_from_store(chunk_id: str) -> Optional[str]:
    """
    Directly read frame_path from store._meta when _frame_index is None.
    This bypasses VisualStoreAdapter entirely — useful when the adapter
    failed to initialise but the store itself is loaded.
    """
    if store is None:
        return None
    try:
        meta_store: Dict[str, Any] = getattr(store, "_meta", {}) or {}
        meta = meta_store.get(chunk_id)
        if meta and meta.get("chunk_type") == "visual":
            chunk_meta = meta.get("metadata", {}) or {}
            return chunk_meta.get("frame_path") or ""
    except Exception:
        pass
    return None


@app.get("/vision/frame/{chunk_id}")
def vision_frame_file(chunk_id: str):
    """
    Serve a keyframe image by chunk_id.

    Resolution strategy (in order):
      1. Lazy-init _frame_index if it is None but store is ready
      2. Look up chunk in the frame index → get stored frame_path
      3. Resolve that path: absolute / relative-to-BASE_DIR / relative-to-CWD
      4. If index unavailable, read frame_path directly from store._meta
      5. Last resort: scan data/frames/** for a file containing chunk_id
      6. 404 with a clear diagnostic message
    """
    global _frame_index

    frame_path_resolved: Optional[str] = None

    # ── Strategy 0: lazy-init frame index if store is ready but index isn't ──
    if _frame_index is None and store is not None:
        try:
            _frame_index = VisualStoreAdapter(store, vision_indexer=vision)
            logger.info("vision/frame: lazy-initialised VisualStoreAdapter for chunk_id=%s", chunk_id)
        except Exception as _e:
            logger.warning("vision/frame: lazy VisualStoreAdapter init failed: %s", _e)

    # ── Strategy 1 + 2: use index metadata ──────────────────────────
    raw_path: str = ""
    if _frame_index is not None and hasattr(_frame_index, "get_frame"):
        frame = _frame_index.get_frame(chunk_id)
        if frame:
            raw_path = frame.get("frame_path") or ""
            frame_path_resolved = _resolve_frame_path(raw_path)
            if frame_path_resolved is None and raw_path:
                logger.warning(
                    "vision/frame: chunk_id=%s found in index but file missing "
                    "(stored path=%r, BASE_DIR=%s)",
                    chunk_id, raw_path, BASE_DIR,
                )

    # ── Strategy 3: read frame_path directly from store._meta ───────
    if frame_path_resolved is None:
        direct_path = _get_frame_path_from_store(chunk_id)
        if direct_path:
            raw_path = direct_path
            frame_path_resolved = _resolve_frame_path(direct_path)
            if frame_path_resolved:
                logger.info(
                    "vision/frame: chunk_id=%s resolved via store._meta at %s",
                    chunk_id, frame_path_resolved,
                )

    # ── Strategy 4: disk scan fallback ──────────────────────────────
    if frame_path_resolved is None:
        frame_path_resolved = _scan_frames_dir_for_chunk(chunk_id)
        if frame_path_resolved:
            logger.info(
                "vision/frame: chunk_id=%s found via disk scan at %s",
                chunk_id, frame_path_resolved,
            )

    # ── 404 with diagnostic ─────────────────────────────────────────
    if frame_path_resolved is None:
        has_index = _frame_index is not None and hasattr(_frame_index, "get_frame")
        logger.warning(
            "vision/frame: 404 for chunk_id=%s | index_available=%s | BASE_DIR=%s",
            chunk_id, has_index, BASE_DIR,
        )
        raise HTTPException(
            404,
            detail=(
                f"Frame not found for chunk_id={chunk_id}. "
                f"Index available: {has_index}. "
                f"Frames root checked: {BASE_DIR / 'data' / 'frames'}"
            ),
        )

    media_type = "image/png" if frame_path_resolved.endswith(".png") else "image/jpeg"
    return FileResponse(
        frame_path_resolved,
        media_type=media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.get("/vision/frames/{video_id}")
def vision_list_frames(video_id: str, limit: int = 50):
    # BUG-FIX: Return empty list (not 503) when vision/frame index is not
    # initialised.  The keyframes gallery simply renders "No keyframes yet"
    # rather than an unhandled server error.
    if _frame_index is None or not hasattr(_frame_index, "list_frames"):
        return {"video_id": video_id, "count": 0, "frames": []}
    frames = [_decorate_visual_result(f) for f in _frame_index.list_frames(video_id=video_id, limit=limit)]
    return {"video_id": video_id, "count": len(frames), "frames": frames}


@app.delete("/vision/frames/{video_id}")
def vision_delete_frames(video_id: str):
    if _frame_index is None or not hasattr(_frame_index, "delete_video"):
        raise HTTPException(503, "frame index not ready")
    removed = _frame_index.delete_video(video_id)
    remaining = len(_frame_index) if hasattr(_frame_index, "__len__") else None
    return {"removed": removed, "remaining": remaining}


# =====================================================================
# BUG-4: New metadata endpoints
# =====================================================================

class VideoMetadataPatch(BaseModel):
    """Fields the user can PATCH on a video's metadata record."""
    title: Optional[str] = None
    summary: Optional[str] = None
    keywords: Optional[List[str]] = None
    topics: Optional[List[str]] = None


@app.get("/videos/search")
async def search_videos(q: str = "", limit: int = 20):
    """
    BUG-4 FIX: Discovery endpoint.
    GET /videos/search?q=machine+learning
    Returns videos whose title/summary/keywords/topics match the query.
    """
    if _metadata_registry is None:
        raise HTTPException(status_code=503, detail="Metadata registry not available")
    results = _metadata_registry.search(query=q, limit=limit)
    return {
        "query": q,
        "count": len(results),
        "videos": [m.to_dict() for m in results],
    }


@app.get("/videos/discover")
async def discover_videos(q: str = "", limit: int = 8, top_k: int = 30):
    """
    SEMANTIC discovery: search chunk-level embeddings across ALL videos,
    then group results by video and return the best-matching videos with
    the most relevant transcript snippets.

    GET /videos/discover?q=machine+learning&limit=8

    Returns:
    {
      "query": "...",
      "count": N,
      "videos": [
        {
          "video_id": "...",
          "title": "...",
          "domain": "...",
          "score": 0.87,          # best chunk score for this video
          "match_count": 3,       # how many chunks matched
          "snippets": [           # up to 3 transcript excerpts
            {"text": "...", "start_time": 12.3, "score": 0.87}
          ],
          "summary": "...",
          "keywords": [...]
        }
      ]
    }
    """
    if not q.strip():
        # Return all videos when query is empty
        all_metas = _metadata_registry.list_all() if _metadata_registry else []
        return {
            "query": q,
            "count": len(all_metas[:limit]),
            "videos": [
                {
                    "video_id": m.video_id,
                    "title": m.title or m.video_id,
                    "domain": m.domain,
                    "score": 0.0,
                    "match_count": 0,
                    "snippets": [],
                    "summary": m.summary,
                    "keywords": m.keywords,
                    "topics": m.topics,
                }
                for m in all_metas[:limit]
            ],
        }

    if store is None or embedder is None:
        # Fallback to metadata keyword search
        metas = (_metadata_registry.search(query=q, limit=limit)
                 if _metadata_registry else [])
        return {
            "query": q,
            "count": len(metas),
            "videos": [
                {
                    "video_id": m.video_id,
                    "title": m.title or m.video_id,
                    "domain": m.domain,
                    "score": 0.5,
                    "match_count": 1,
                    "snippets": [],
                    "summary": m.summary,
                    "keywords": m.keywords,
                    "topics": m.topics,
                }
                for m in metas
            ],
        }

    loop = asyncio.get_running_loop()
    # Search across ALL videos (video_id=None)
    raw_results: List[DotDict] = await loop.run_in_executor(
        _search_executor,
        _do_search_sync,
        q, None, top_k, False,
    )

    # Group chunks by video_id, keep best score + collect snippets
    from collections import defaultdict
    grouped: dict = defaultdict(lambda: {"score": 0.0, "snippets": [], "match_count": 0})

    for chunk in raw_results:
        vid = chunk.get("video_id") or chunk.get("id", "").split("_")[0]
        if not vid:
            continue
        score = float(chunk.get("score", chunk.get("dense_score", 0.0)))
        grouped[vid]["match_count"] += 1
        if score > grouped[vid]["score"]:
            grouped[vid]["score"] = score
        # Keep up to 3 snippet excerpts
        if len(grouped[vid]["snippets"]) < 3:
            text = (chunk.get("text") or "").strip()
            if text:
                grouped[vid]["snippets"].append({
                    "text": text[:200],
                    "start_time": chunk.get("start_time"),
                    "score": round(score, 3),
                })

    # Sort videos by best-chunk score descending
    sorted_videos = sorted(grouped.items(), key=lambda x: -x[1]["score"])

    # Enrich with metadata
    output = []
    for vid, data in sorted_videos[:limit]:
        meta = _metadata_registry.get(vid) if _metadata_registry else None
        output.append({
            "video_id": vid,
            "title": (meta.title if meta else None) or vid,
            "domain": (meta.domain if meta else None) or "",
            "score": round(data["score"], 3),
            "match_count": data["match_count"],
            "snippets": data["snippets"],
            "summary": (meta.summary if meta else None) or "",
            "keywords": (meta.keywords if meta else None) or [],
            "topics": (meta.topics if meta else None) or [],
        })

    return {"query": q, "count": len(output), "videos": output}


@app.get("/video/{video_id}/chapters")
async def get_video_chapters(video_id: str):
    """
    Build chapters from indexed speech chunks by grouping them into
    time windows (~60 s each). Each chapter gets:
      - index, title (from topics/keywords or auto-generated),
        start_time, end_time, summary (first ~200 chars of text),
        thumbnail_url (nearest visual frame if available).
    Falls back gracefully when chunks / frames aren't available.
    Chapters are cached in VideoMetadata.extra["chapters"] so they are
    built only once and returned instantly on subsequent requests.
    """
    if store is None:
        return {"video_id": video_id, "chapters": []}

    # ── 0. Return cached chapters if already built ───────────────────
    if _metadata_registry is not None:
        _cached_meta = _metadata_registry.get(video_id)
        if _cached_meta is not None:
            _cached_chapters = _cached_meta.extra.get("chapters")
            if _cached_chapters:
                return {"video_id": video_id, "chapters": _cached_chapters, "cached": True}

    # ── 1. Collect all speech chunks sorted by time ──────────────────
    raw_chunks: List[Any] = []

    # Primary: use get_all_chunks_for_video (no embedding required)
    fn = getattr(store, "get_all_chunks_for_video", None)
    if callable(fn):
        try:
            raw_chunks = fn(video_id=video_id, speech_only=False) or []
        except Exception:
            pass

    # Secondary: try legacy method names some store implementations may expose
    if not raw_chunks:
        for method in ["get_chunks", "list_chunks", "get_video_chunks"]:
            fn = getattr(store, method, None)
            if callable(fn):
                try:
                    raw_chunks = fn(video_id=video_id) or []
                    if raw_chunks:
                        break
                except Exception:
                    pass

    speech_chunks = []
    for c in raw_chunks:
        ct = (c.get("chunk_type") if isinstance(c, dict) else getattr(c, "chunk_type", "speech")) or "speech"
        if ct == "visual":
            continue
        st = float((c.get("start_time") if isinstance(c, dict) else getattr(c, "start_time", 0)) or 0)
        et = float((c.get("end_time")   if isinstance(c, dict) else getattr(c, "end_time",   st + 30)) or st + 30)
        tx = (c.get("text") or c.get("content")) if isinstance(c, dict) else (getattr(c, "text", "") or getattr(c, "content", ""))
        speech_chunks.append({"start": st, "end": et, "text": (tx or "").strip()})

    speech_chunks.sort(key=lambda x: x["start"])

    if not speech_chunks:
        return {"video_id": video_id, "chapters": []}

    # ── 2. Group into ~60-second windows ────────────────────────────
    WINDOW = 60.0
    groups: List[dict] = []
    cur: Optional[dict] = None
    for chunk in speech_chunks:
        if cur is None or chunk["start"] >= cur["start"] + WINDOW:
            if cur:
                groups.append(cur)
            cur = {"start": chunk["start"], "end": chunk["end"], "texts": [chunk["text"]]}
        else:
            cur["end"] = max(cur["end"], chunk["end"])
            if chunk["text"]:
                cur["texts"].append(chunk["text"])
    if cur:
        groups.append(cur)

    # ── 3. Pull metadata topics for chapter titles ───────────────────
    meta_topics: List[str] = []
    if _metadata_registry is not None:
        m = _metadata_registry.get(video_id)
        if m:
            meta_topics = (m.topics or m.keywords or [])

    # ── 4. Pull visual frames for thumbnails ─────────────────────────
    visual_frames: List[dict] = []
    if _frame_index is not None and hasattr(_frame_index, "list_frames"):
        try:
            visual_frames = _frame_index.list_frames(video_id=video_id, limit=200)
        except Exception:
            pass
    visual_frames.sort(key=lambda f: float(f.get("timestamp") or 0))

    def nearest_frame_url(t: float) -> Optional[str]:
        """Return the thumbnail_url of the visual frame closest to time t."""
        if not visual_frames:
            return None
        best = min(visual_frames, key=lambda f: abs(float(f.get("timestamp") or 0) - t))
        cid = best.get("chunk_id", "")
        return _public_frame_url(cid) if cid else None

    # ── 5. Build chapter objects ─────────────────────────────────────
    chapters = []
    for i, g in enumerate(groups):
        topic = meta_topics[i] if i < len(meta_topics) else None
        combined_text = " ".join(g["texts"])
        if not topic:
            # Auto-title: first 60 chars of the chunk text
            topic = combined_text[:60].rstrip() + ("…" if len(combined_text) > 60 else "")
        summary_text = combined_text[:200].rstrip() + ("…" if len(combined_text) > 200 else "")
        chapters.append({
            "index": i + 1,
            "title": topic or f"Chapter {i + 1}",
            "start_time": g["start"],
            "end_time": g["end"],
            "summary": summary_text,
            "thumbnail_url": nearest_frame_url(g["start"]),
        })

    # ── 6. Persist chapters into metadata for instant future loads ───
    if _metadata_registry is not None and chapters:
        _meta_to_update = _metadata_registry.get(video_id)
        if _meta_to_update is not None:
            _meta_to_update.extra["chapters"] = chapters
            _meta_to_update.touch()
            _metadata_registry.register(_meta_to_update)

    return {"video_id": video_id, "chapters": chapters}


@app.get("/video/{video_id}/metadata")
async def get_video_metadata(video_id: str):
    """
    BUG-4 FIX: Return the full VideoMetadata record for a video.
    """
    if _metadata_registry is None:
        raise HTTPException(status_code=503, detail="Metadata registry not available")
    meta = _metadata_registry.get(video_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"No metadata for video_id={video_id!r}")
    return meta.to_dict()


@app.patch("/video/{video_id}/metadata")
async def patch_video_metadata(video_id: str, patch: VideoMetadataPatch):
    """
    BUG-4 FIX: Update title / summary / keywords / topics for a video.
    Useful when the auto-generated title/summary is wrong.
    """
    if _metadata_registry is None:
        raise HTTPException(status_code=503, detail="Metadata registry not available")
    meta = _metadata_registry.get(video_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"No metadata for video_id={video_id!r}")
    if patch.title is not None:
        meta.title = patch.title
    if patch.summary is not None:
        meta.summary = patch.summary
    if patch.keywords is not None:
        meta.keywords = patch.keywords
    if patch.topics is not None:
        meta.topics = patch.topics
    meta.touch()
    _metadata_registry.register(meta)
    logger.info("Patched metadata for %s: %s", video_id, patch.model_dump(exclude_none=True))
    return meta.to_dict()


@app.post("/video/{video_id}/metadata/regenerate")
async def regenerate_video_metadata(video_id: str):
    """
    BUG-4 FIX: Re-run LLM summarization for a video that already exists in the store.
    Useful when the LLM was unavailable at ingestion time.
    """
    if _metadata_registry is None:
        raise HTTPException(status_code=503, detail="Metadata registry not available")
    if store is None:
        raise HTTPException(status_code=503, detail="Store not available")

    # Fetch chunks from the store to regenerate metadata
    loop = asyncio.get_running_loop()
    try:
        chunks = await loop.run_in_executor(
            _search_executor,
            lambda: _fetch_chunks_for_video(video_id),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch chunks: {e}")

    if not chunks:
        raise HTTPException(status_code=404,
                            detail=f"No chunks found for video_id={video_id!r}")

    meta = _metadata_registry.generate_and_register(
        video_id=video_id,
        chunks=chunks,
        title="",
        llm_client=llm_client,
        ollama_url=getattr(settings, "ollama_base_url", "http://localhost:11434"),
        ollama_model=getattr(settings, "ollama_chat_model", "qwen3.5:2b"),
    )
    return meta.to_dict()


def _get_video_metadata_dict(video_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    v7.0: Fetch video metadata for LLM context injection.
    Returns a dict with title, summary, domain, keywords, topics, language.
    Also surfaces chapters from extra to top-level so llm_client can inject them.
    """
    if not video_id or _metadata_registry is None:
        return None
    try:
        meta = _metadata_registry.get(video_id)
        if meta is None:
            return None
        d = meta.to_dict()
        # Surface chapters from extra so _build_video_context_block can use them
        if "chapters" not in d and isinstance(d.get("extra"), dict):
            chapters = d["extra"].get("chapters")
            if chapters:
                d["chapters"] = chapters
        return d
    except Exception as e:
        logger.warning("_get_video_metadata_dict(%s) failed: %s", video_id, e)
        return None


def _fetch_chunks_for_video(video_id: str) -> List[Any]:
    """Retrieve stored chunks for a video from the store (sync)."""
    if store is None:
        return []
    # Try store-specific method names
    for method in ["get_chunks", "list_chunks", "get_video_chunks"]:
        fn = getattr(store, method, None)
        if callable(fn):
            try:
                return fn(video_id=video_id) or []
            except Exception:
                pass
    return []


# =====================================================================
# CACHE endpoints
# =====================================================================

@app.get("/cache/stats")
async def cache_stats():
    """
    Return statistics about the semantic query cache:
    total entries, total hits, top cached questions, etc.
    """
    if _query_cache is None:
        return {"enabled": False, "reason": "QueryCache not initialised"}
    return _query_cache.stats()


@app.delete("/cache/video/{video_id}")
async def cache_clear_video(video_id: str):
    """
    Remove all cached query/answer pairs for a specific video.
    Call this if the video was re-indexed and the old answers are stale.
    """
    if _query_cache is None:
        raise HTTPException(status_code=503, detail="QueryCache not available")
    n = _query_cache.delete_video(video_id)
    return {"video_id": video_id, "deleted_entries": n}


@app.delete("/cache/all")
async def cache_clear_all():
    """
    Wipe the entire query cache.
    Useful after a full re-index of all videos.
    """
    if _query_cache is None:
        raise HTTPException(status_code=503, detail="QueryCache not available")
    # Re-create the table (fastest wipe)
    import sqlite3 as _sq
    db_path = _query_cache._db_path
    with _sq.connect(db_path) as conn:
        conn.execute("DELETE FROM query_cache")
        conn.commit()
        n = conn.execute("SELECT changes()").fetchone()[0]
    logger.info("QueryCache wiped (%d entries removed)", n)
    return {"deleted_entries": n, "status": "cache cleared"}


# ── Global shared history ────────────────────────────────────────────

_HISTORY_FILE = BASE_DIR / "data" / "faiss_store" / "global_history.json"
_HISTORY_MAX  = 500          # keep at most this many entries
_HISTORY_TTL  = 3 * 24 * 3600  # 3 days in seconds
_history_lock = __import__("asyncio").Lock()

def _history_load() -> list:
    try:
        if _HISTORY_FILE.exists():
            return __import__("json").loads(_HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return []

def _history_save(items: list) -> None:
    _HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    _HISTORY_FILE.write_text(__import__("json").dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

def _history_purge(items: list) -> list:
    cutoff = __import__("time").time() - _HISTORY_TTL
    return [i for i in items if i.get("ts", 0) > cutoff]


@app.get("/history")
async def history_list():
    """Return all global history entries (TTL-purged)."""
    async with _history_lock:
        items = _history_purge(_history_load())
        _history_save(items)
    return {"entries": items}


@app.post("/history")
async def history_add(payload: dict):
    """Append one entry to the global history."""
    import time, uuid
    entry = {
        "id":         str(uuid.uuid4()),
        "ts":         time.time(),
        "videoId":    payload.get("videoId", ""),
        "videoTitle": payload.get("videoTitle", ""),
        "question":   payload.get("question", ""),
        "answer":     payload.get("answer", ""),
    }
    async with _history_lock:
        items = _history_purge(_history_load())
        items.insert(0, entry)
        if len(items) > _HISTORY_MAX:
            items = items[:_HISTORY_MAX]
        _history_save(items)
    return {"status": "ok", "entry": entry}


@app.delete("/history")
async def history_clear():
    """Wipe all global history."""
    async with _history_lock:
        _history_save([])
    return {"status": "ok"}


@app.delete("/history/{entry_id}")
async def history_delete_one(entry_id: str):
    """Delete a single history entry by id."""
    async with _history_lock:
        items = [i for i in _history_load() if i.get("id") != entry_id]
        _history_save(items)
    return {"status": "ok"}


# ── HTML fallback ────────────────────────────────────────────────────

HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>Video Intelligence API</title>
    <meta charset="utf-8">
    <style>
        body { font-family: sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #333; }
        .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
        code { background: #e0e0e0; padding: 2px 5px; border-radius: 3px; }
    </style>
</head>
<body>
    <h1>🎬 Video Intelligence API v6.2</h1>
    <p>Video understanding with transcription, vision (VLM/CLIP), search, and LLM synthesis.</p>
    <h2>📡 Key Endpoints</h2>
    <div class="endpoint">
        <strong>POST /ask</strong> — BUG-7: Man-in-the-loop clarification<br>
        <code>{ "question": "...", "confirmed": false }</code> → returns clarification<br>
        <code>{ "question": "...", "confirmed": true, "reformulated_question": "..." }</code> → full answer
    </div>
    <div class="endpoint"><strong>GET /videos/search?q=...</strong> — BUG-4: Discovery search (returns matching videos)</div>
    <div class="endpoint"><strong>GET /video/{id}/metadata</strong> — Full metadata record</div>
    <div class="endpoint"><strong>PATCH /video/{id}/metadata</strong> — Update title/summary/keywords</div>
    <div class="endpoint"><strong>POST /video/{id}/metadata/regenerate</strong> — Re-run LLM summarization</div>
    <div class="endpoint"><strong>POST /upload</strong> — Upload video file</div>
    <div class="endpoint"><strong>POST /search</strong> — Semantic search</div>
    <div class="endpoint"><strong>GET /videos</strong> — List indexed videos</div>
    <div class="endpoint"><strong>GET /health</strong> — Service health check</div>
    <hr>
    <p><a href="/docs">📚 Swagger UI</a> | <a href="/redoc">📖 ReDoc</a></p>
</body>
</html>
"""


# =====================================================================
# MARKETING SITE (project/) — mounted at the site ROOT.
#
# MUST be the LAST route registered: a mount at "/" matches every path as a
# prefix, so it is only reached for URLs that no explicit API route or earlier
# mount (/static, /app-assets, /app, /videos, /ask, …) already handled.
# This lets the landing page served at "/" (project/VGIA.html) load its
# relative assets (/assets/*.png) and navigate to its sibling pages
# (/login.html, /signup.html, /pricing.html, /documentation.html, …).
# =====================================================================
if PROJECT_DIR.exists():
    app.mount("/", StaticFiles(directory=str(PROJECT_DIR), html=True), name="site")
    logger.info("✅ Marketing site mounted at / (project/)")