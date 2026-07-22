"""
core/graph_search.py — High-level GraphRAG query interface (v1.0)
==================================================================
Glue layer between the existing hybrid retriever (FAISS + BM25 + RRF +
cross-encoder + HyDE) and the new knowledge-graph store.

Public API
----------
    gs = GraphSearch(
        graph_store=store,           # core.graph_store.GraphStore
        vector_store=video_store,    # core.storage.VideoStore (existing)
        embedder=embedder,           # core.embeddings.EmbeddingGenerator
        reranker=reranker,           # core.reranker.CrossEncoderReranker
        llm_client=llm_client,       # module core.llm_client OR object
    )

    out = gs.ask(
        question="What did Alice say about BB84?",
        mode="auto",                 # "auto" | "local" | "global" | "hybrid" | "vector"
        video_id=None,
        top_k=8,
    )

    out = {
        "mode_used": "hybrid",
        "answer": "...",
        "graph": {... GraphSearchResult.as_dict() ...},
        "vector": [...SearchResult dicts...],
        "citations": [...],
        "elapsed_seconds": 4.21,
    }

Design
------
* `mode="auto"`  picks the strategy from a tiny intent classifier:
    - global   → "main themes", "summarize", "key topics", "overview"
    - local    → "what did X say about Y", "who is X", "where is Y"
    - hybrid   → everything else (default & safest)

* In hybrid mode: both LOCAL graph context AND vector hits are gathered,
  the LLM is given BOTH so it can reason across entities AND quote chunks
  with timestamps.

Author : Spiral father, 2026.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Safe settings
# ─────────────────────────────────────────────────────────────────────
def _cfg(name: str, default: Any) -> Any:
    try:
        from config.settings import settings  # type: ignore
        return getattr(settings, name, default)
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────
# Intent detection (rule-based, multilingual ar/en/fr)
# ─────────────────────────────────────────────────────────────────────
GLOBAL_KEYWORDS = {
    "main themes", "main topics", "overview", "summary", "summarize",
    "key ideas", "key topics", "in summary", "what is this about",
    "résumé", "résume", "thèmes principaux", "idées principales",
    "sujets principaux", "vue d'ensemble", "synthèse",
    "ملخص", "المواضيع", "الأفكار الرئيسية", "ملخّص",
}
LOCAL_KEYWORDS_RE = re.compile(
    r"\b(who|what|where|when|how|which|tell me about|did .* say|qui|quoi|où|quand|comment|من|ماذا|أين|متى|كيف)\b",
    re.IGNORECASE,
)


def detect_intent(question: str) -> str:
    """Returns 'global', 'local', 'hybrid', or 'timestamp_lookup'."""
    if not question:
        return "hybrid"
    q_low = question.lower().strip()
    # Timestamp questions: delegate to llm_client's extractor via import
    try:
        from core.llm_client import _extract_timestamp_seconds
        if _extract_timestamp_seconds(question) is not None:
            return "timestamp_lookup"
    except Exception:
        pass
    if any(kw in q_low for kw in GLOBAL_KEYWORDS):
        return "global"
    # capitalized name → likely local
    has_capitalized = bool(re.search(r"\b[A-ZÀ-Ý][a-zà-ÿ]{2,}\b", question))
    if LOCAL_KEYWORDS_RE.search(question) and (has_capitalized or len(question.split()) <= 12):
        return "local"
    return "hybrid"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "as_dict"):
        try:
            return obj.as_dict()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    return {"value": str(obj)}


def _fmt_ts(seconds: Any) -> str:
    try:
        s = int(float(seconds))
    except Exception:
        return "00:00:00"
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ─────────────────────────────────────────────────────────────────────
# GraphSearch
# ─────────────────────────────────────────────────────────────────────
@dataclass
class GraphSearchConfig:
    default_mode: str = "auto"
    local_hops: int = 1
    local_max_nodes: int = 25
    global_top_k: int = 3
    vector_top_k: int = 8
    rerank_top_k: int = 6
    use_reranker: bool = True
    max_context_chars: int = 6500
    llm_num_predict: int = 384
    llm_temperature: float = 0.25


class GraphSearch:
    """
    High-level orchestrator that combines the knowledge graph and the
    existing hybrid vector retriever to answer questions.
    """

    def __init__(
        self,
        graph_store: Any,
        vector_store: Any = None,
        embedder: Any = None,
        reranker: Any = None,
        llm_client: Any = None,
        config: Optional[GraphSearchConfig] = None,
    ):
        self.graph = graph_store
        self.vstore = vector_store
        self.embedder = embedder
        self.reranker = reranker
        self.llm = llm_client
        self.cfg = config or GraphSearchConfig()
        logger.info(
            "GraphSearch v1.0 | graph=%s | vstore=%s | reranker=%s | llm=%s",
            self.graph is not None,
            self.vstore is not None,
            self.reranker is not None,
            self.llm is not None,
        )

    # ─────────────────────────────────────────────────────────────────
    # ENTRY POINT
    # ─────────────────────────────────────────────────────────────────
    def ask(
        self,
        question: str,
        mode: str = "auto",
        video_id: Optional[str] = None,
        top_k: Optional[int] = None,
        language: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """End-to-end: question → graph + vector → LLM → answer."""
        t0 = time.time()
        question = (question or "").strip()
        if not question:
            return self._empty(t0, "empty question")

        intent = mode if mode != "auto" else detect_intent(question)
        logger.info("GraphSearch.ask | intent=%s | video_id=%s", intent, video_id)

        # ── FAST PATH: speaker-count questions ───────────────────────────────
        # Answer directly from graph PERSON nodes — no LLM call needed.
        _SPEAKER_COUNT_RE = re.compile(
            r"\b(how many|number of|count of|combien|كم)\b"
            r".*"
            r"\b(speaker|speakers|person|people|voice|voices|متحدث|متحدثون|متحدثين|شخص|أشخاص|صوت|أصوات)\b",
            re.IGNORECASE,
        )
        # Also catch Arabic-first patterns: "كم من متحدث", "كم شخص يتحدث", "عدد المتحدثين"
        _SPEAKER_COUNT_AR_RE = re.compile(
            r"(كم\s*(من\s*)?(متحدث|متحدثون|متحدثين|شخص|أشخاص|صوت|أصوات)|"
            r"عدد\s*(المتحدثين|الأشخاص|الأصوات)|"
            r"(متحدث|شخص|صوت)\s*في\s*الفيديو)",
            re.IGNORECASE,
        )
        # Detect language of question for the answer
        _q_ar_chars = sum(1 for c in question if "\u0600" <= c <= "\u06FF")
        _q_is_arabic = _q_ar_chars >= max(1, len(question.replace(" ", "")) * 0.15)

        if (_SPEAKER_COUNT_RE.search(question) or _SPEAKER_COUNT_AR_RE.search(question)) \
                and self.graph is not None and video_id:
            try:
                speakers = self.graph.get_speakers_for_video(video_id)
                n = len(speakers)
                names = [s["name"] for s in speakers]
                if _q_is_arabic:
                    if n == 0:
                        answer = ("لم يتم التعرف على أي متحدث في هذا الفيديو. "
                                  "جرّب إعادة الفهرسة لتحليل المتحدثين.")
                    elif n == 1:
                        answer = f"يوجد **متحدث واحد** في هذا الفيديو: {names[0]}."
                    else:
                        formatted = "، ".join(f"**{nm}**" for nm in names[:-1]) + f" و**{names[-1]}**"
                        answer = f"يوجد **{n} متحدثين** في هذا الفيديو: {formatted}."
                else:
                    if n == 0:
                        answer = ("No speakers were identified in the graph for this video. "
                                  "Try calling POST /graph/enrich/{video_id} to populate speaker nodes.")
                    elif n == 1:
                        answer = f"There is **1 speaker** in this video: {names[0]}."
                    else:
                        formatted = ", ".join(f"**{nm}**" for nm in names[:-1]) + f" and **{names[-1]}**"
                        answer = f"There are **{n} speakers** in this video: {formatted}."
                logger.info("speaker-count fast-path: %d speakers for video=%s", n, video_id)
                return {
                    "answer": answer,
                    "intent": "factual",
                    "mode": "speaker_count",
                    "graph_nodes": n,
                    "vec_hits": 0,
                    "elapsed_s": round(time.time() - t0, 2),
                    "sources": [],
                    "graph_context": "",
                }
            except Exception as _spk_err:
                logger.warning("speaker-count fast-path failed: %s", _spk_err)
        # ────────────────────────────────────────────────────────────────────

        graph_res = None
        if self.graph is not None:
            try:
                if intent == "global":
                    graph_res = self.graph.global_search(
                        question, top_k=self.cfg.global_top_k, video_id=video_id,
                    )
                elif intent == "local":
                    graph_res = self.graph.local_search(
                        query=question,
                        hops=self.cfg.local_hops,
                        max_nodes=self.cfg.local_max_nodes,
                        video_id=video_id,
                    )
                else:  # hybrid, timestamp_lookup, yes_no, etc.
                    graph_res = self.graph.local_search(
                        query=question,
                        hops=self.cfg.local_hops,
                        max_nodes=self.cfg.local_max_nodes,
                        video_id=video_id,
                    )
            except Exception as exc:
                logger.warning("Graph search failed: %s", exc)
                graph_res = None

        # ── Retrieval strategy ──────────────────────────────────────────
        # timestamp_lookup → time-range scan (exact) + semantic fallback merged
        # yes_no           → full transcript scan + semantic hits merged
        # global           → skip vector search (graph-only)
        # everything else  → normal semantic search
        vector_hits: List[Any] = []
        if intent != "global" and self.vstore is not None:

            if intent == "timestamp_lookup" and video_id:
                # Pull exact time-range chunks directly from the store
                try:
                    from core.llm_client import _extract_timestamp_seconds
                    target_ts = _extract_timestamp_seconds(question)
                    if target_ts is not None:
                        # Tight primary window around the requested timestamp
                        window = 30.0
                        time_hits = self.vstore.get_chunks_by_time_range(
                            video_id=video_id,
                            start_seconds=max(0.0, target_ts - window),
                            end_seconds=target_ts + window,
                        )
                        # Sort time_hits by proximity to target (closest first)
                        def _proximity(r):
                            s = float(getattr(r, "start_time", None) or 0)
                            return abs(s - target_ts)
                        time_hits.sort(key=_proximity)
                        logger.info(
                            "timestamp_lookup: target=%.1fs window=%.1fs → %d time-range chunks",
                            target_ts, window, len(time_hits),
                        )
                        # Semantic hits: only add if they are also near the target
                        # (within 2× window) — prevents distant chunks polluting results
                        semantic_hits = self._vector_search(
                            question, video_id=video_id,
                            top_k=top_k or self.cfg.vector_top_k,
                            language=language,
                        )
                        seen = {getattr(h, "chunk_id", None) for h in time_hits}
                        for h in semantic_hits:
                            cid = getattr(h, "chunk_id", None)
                            if cid in seen:
                                continue
                            h_start = float(getattr(h, "start_time", None) or 0)
                            # Only merge semantic hits that are near the target time
                            if abs(h_start - target_ts) <= window * 2:
                                time_hits.append(h)
                                seen.add(cid)
                        vector_hits = time_hits
                    else:
                        vector_hits = self._vector_search(
                            question, video_id=video_id,
                            top_k=top_k or self.cfg.vector_top_k,
                            language=language,
                        )
                except Exception as exc:
                    logger.warning("timestamp_lookup retrieval failed: %s", exc)
                    vector_hits = self._vector_search(
                        question, video_id=video_id,
                        top_k=top_k or self.cfg.vector_top_k,
                        language=language,
                    )

            elif intent == "yes_no" and video_id:
                # Semantic hits only — full transcript scan adds latency without
                # meaningful quality gain for yes/no questions. Cap at top-12.
                try:
                    vector_hits = self._vector_search(
                        question, video_id=video_id,
                        top_k=max(top_k or self.cfg.vector_top_k, 12),
                        language=language,
                    )
                    logger.info("yes_no semantic search: %d hits", len(vector_hits))
                except Exception as exc:
                    logger.warning("yes_no search failed: %s", exc)
                    vector_hits = []
                # Rerank only if result set is small enough (≤ 8) to stay fast
                if self.cfg.use_reranker and self.reranker is not None and 0 < len(vector_hits) <= 8:
                    vector_hits = self._safe_rerank(question, vector_hits)

            else:
                # Normal hybrid semantic search.
                # NOTE: results coming from VideoStore.search() have already been
                # reranked by _do_search_sync in main.py (cross-encoder, ~18-24 s).
                # Running the reranker a second time here gives no quality gain and
                # doubles the CPU cost — skip it.
                effective_top_k = top_k or self.cfg.vector_top_k
                vector_hits = self._vector_search(
                    question, video_id=video_id,
                    top_k=effective_top_k,
                    language=language,
                )

        # LLM synthesis
        # Skip internal LLM synthesis — the main stream endpoint already injects
        # graph context into its own LLM call. Running a second full LLM here
        # added ~40-55s of latency for zero quality gain.
        # We only need: graph context_text (for the stream prompt) + vector hits.
        citations = self._build_citations(vector_hits, graph_res)
        answer = ""
        model_used = "none"

        elapsed = time.time() - t0
        logger.info(
            "GraphSearch.ask | mode=%s | graph_nodes=%d | vec_hits=%d | %.2fs",
            intent,
            len(graph_res.nodes) if graph_res else 0,
            len(vector_hits),
            elapsed,
        )

        return {
            "question": question,
            "mode_used": intent,
            "answer": answer,
            "model_used": model_used,
            "graph": graph_res.as_dict() if graph_res else None,
            "vector": [self._normalize_hit(h) for h in vector_hits],
            "citations": citations,
            "elapsed_seconds": round(elapsed, 2),
            "video_id": video_id,
        }

    # ─────────────────────────────────────────────────────────────────
    # Vector search (delegates to existing VideoStore)
    # ─────────────────────────────────────────────────────────────────
    def _vector_search(
        self, question: str, video_id: Optional[str], top_k: int,
        language: Optional[str] = None,
    ) -> List[Any]:
        if self.vstore is None:
            return []
        try:
            qe = None
            if self.embedder is not None:
                # try the typical embedder API surface
                for fn_name in ("encode_query", "embed_query", "encode"):
                    fn = getattr(self.embedder, fn_name, None)
                    if callable(fn):
                        try:
                            # Pass language so HyDE generates hypothesis in the right language
                            import inspect
                            if "language" in inspect.signature(fn).parameters:
                                qe = fn(question, language=language)
                            else:
                                qe = fn(question)
                            break
                        except Exception:
                            pass
            search_kwargs = {
                "query_embedding": qe,
                "query": question,
                "limit": top_k,
                "top_k": top_k,
                "video_id": video_id,
                "with_vision": True,
                "language": language,   # hints BM25/FAISS lang-shard routing
            }
            return self.vstore.search(**search_kwargs) or []
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []

    def _ask_multi_hop(
        self,
        question: str,
        video_id: Optional[str] = None,
        top_k: int = 6,
        language: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Multi-hop reasoning pipeline.
          1. Parse question → relation types + traversal direction.
          2. Walk knowledge graph hop-by-hop (multi_hop_graph_search).
          3. Merge graph chunk_ids with vector retrieval.
          4. Timeline-aware reranking: graph-anchored chunks first, in
             temporal order; remaining chunks sorted by start_time.
          5. Synthesize from the ordered evidence chain.
        """
        t0 = time.time()
        q_low = question.lower()

        # ── Infer relation filter + direction from question ──────────
        rel_filter = None
        direction = "both"
        if any(w in q_low for w in (
            "after", "following", "then", "next", "subsequently",
            "après", "ensuite", "par la suite", "بعد", "ثم", "بعدها",
        )):
            rel_filter = ["precedes", "causes", "discusses", "part_of"]
            direction = "forward"
        elif any(w in q_low for w in (
            "before", "prior", "previously", "led to", "caused",
            "avant", "auparavant", "قبل", "أدى إلى", "تسبب",
        )):
            rel_filter = ["precedes", "causes", "supports", "contradicts"]
            direction = "backward"
        elif any(w in q_low for w in (
            "cause", "result", "consequence", "because", "therefore",
            "causé", "résultat", "conséquence", "parce que", "donc",
            "سبب", "نتيجة", "لأن", "لذلك",
        )):
            rel_filter = ["causes", "supports", "contradicts", "quantifies"]
        elif any(w in q_low for w in (
            "define", "explain", "mean", "what is", "how does",
            "définir", "expliquer", "qu'est-ce", "ما هو", "ما معنى", "كيف",
        )):
            rel_filter = ["defined_as", "discusses", "exemplifies", "part_of"]

        # ── Graph multi-hop traversal ────────────────────────────────
        graph_result = None
        multi_hop_chunk_ids: List[str] = []
        if self.gstore is not None:
            try:
                graph_result = self.gstore.multi_hop_graph_search(
                    query=question,
                    relation_filter=rel_filter,
                    hops=3,
                    max_nodes_per_hop=8,
                    video_id=video_id,
                    direction=direction,
                    temporal_order=True,
                )
                multi_hop_chunk_ids = graph_result.chunk_ids[:30]
                logger.info(
                    "multi_hop: %d nodes %d edges %d chunks dir=%s rel=%s",
                    len(graph_result.nodes), len(graph_result.edges),
                    len(multi_hop_chunk_ids), direction, rel_filter,
                )
            except Exception as exc:
                logger.warning("multi_hop_graph_search failed: %s", exc)

        # ── Vector retrieval ─────────────────────────────────────────
        top_k_vec = max(top_k * 2, 12)
        vector_hits = self._safe_vector_search(question, video_id=video_id, top_k=top_k_vec)

        # ── Timeline-aware reranking ─────────────────────────────────
        # Chunks in graph chain → front, sorted by chain position (= temporal order).
        # Remaining chunks → sorted by start_time.
        if multi_hop_chunk_ids:
            chain_order = {cid: i for i, cid in enumerate(multi_hop_chunk_ids)}

            def _tkey(hit: Any) -> tuple:
                cid = (
                    getattr(hit, "chunk_id", None)
                    or (hit.get("chunk_id") if isinstance(hit, dict) else None)
                )
                if cid and cid in chain_order:
                    return (0, float(chain_order[cid]))
                try:
                    ts = float(
                        getattr(hit, "start_time", None)
                        or (hit.get("start_time") if isinstance(hit, dict) else 0.0)
                        or 0.0
                    )
                    return (1, ts)
                except Exception:
                    return (2, 0.0)

            vector_hits = sorted(vector_hits, key=_tkey)[:top_k]
            logger.info("Timeline-aware rerank: %d hits ordered by graph chain + time", len(vector_hits))
        else:
            vector_hits = self._safe_rerank(question, vector_hits)[:top_k]

        # ── Load graph-anchored chunks missing from vector results ───
        if multi_hop_chunk_ids and self.vstore is not None:
            existing_ids = {
                getattr(h, "chunk_id", None)
                or (h.get("chunk_id") if isinstance(h, dict) else None)
                for h in vector_hits
            }
            missing = [cid for cid in multi_hop_chunk_ids[:6] if cid not in existing_ids]
            if missing:
                for fn_name in ("get_chunks_by_ids", "fetch_chunks", "get_by_ids"):
                    fn = getattr(self.vstore, fn_name, None)
                    if callable(fn):
                        try:
                            extra = fn(missing) or []
                            vector_hits = list(vector_hits) + list(extra)
                            logger.info("Loaded %d extra graph-anchored chunks", len(extra))
                        except Exception as exc:
                            logger.debug("fetch extra chunks failed: %s", exc)
                        break

        # ── Synthesize ───────────────────────────────────────────────
        answer, citations, model = self._synthesize(
            question=question,
            graph_res=graph_result,
            vector_hits=vector_hits,
            language=language,
            intent="multi_hop",
        )
        return {
            "mode_used": "multi_hop",
            "answer": answer,
            "graph": graph_result.as_dict() if graph_result else {},
            "vector": vector_hits,
            "citations": citations,
            "elapsed_seconds": round(time.time() - t0, 2),
        }

    def _safe_rerank(self, question: str, hits: List[Any]) -> List[Any]:
        try:
            for fn_name in ("rerank", "rerank_results", "score"):
                fn = getattr(self.reranker, fn_name, None)
                if callable(fn):
                    out = fn(query=question, results=hits, top_k=self.cfg.rerank_top_k)
                    if out:
                        return out
        except Exception as exc:
            logger.warning("Rerank failed: %s", exc)
        return hits[: self.cfg.rerank_top_k]

    # ─────────────────────────────────────────────────────────────────
    # LLM synthesis
    # ─────────────────────────────────────────────────────────────────
    def _synthesize(
        self,
        question: str,
        graph_res: Any,
        vector_hits: List[Any],
        language: Optional[str],
        intent: str,
    ) -> Tuple[str, List[Dict[str, Any]], str]:
        # Build context: GRAPH first (entities, relations, communities), then CHUNKS
        ctx_parts: List[str] = []
        if graph_res is not None and getattr(graph_res, "context_text", ""):
            ctx_parts.append(graph_res.context_text)
        if vector_hits:
            ctx_parts.append(self._format_chunks(vector_hits))
        full_ctx = "\n\n".join(ctx_parts).strip()
        if len(full_ctx) > self.cfg.max_context_chars:
            full_ctx = full_ctx[: self.cfg.max_context_chars] + "\n…[truncated]"

        citations = self._build_citations(vector_hits, graph_res)

        # If we have a real llm_client (module from core.llm_client), use it
        try:
            from core import llm_client as lc
            res = lc.synthesize_answer(
                question=question,
                results=vector_hits,
                language=language or "en",
                graph_context=full_ctx,           # picked up by patched llm_client
                num_predict=self.cfg.llm_num_predict,
                temperature=self.cfg.llm_temperature,
                max_items=self.cfg.rerank_top_k,
            )
            answer = (
                getattr(res, "answer", None)
                or (res.get("answer") if isinstance(res, dict) else "")
                or ""
            )
            model = (
                getattr(res, "model", None)
                or (res.get("model") if isinstance(res, dict) else "")
                or "ollama"
            )
            if answer:
                return answer, citations, model
        except Exception as exc:
            logger.debug("llm_client.synthesize_answer unavailable (%s) → fallback", exc)

        # Generic fallback: build a single prompt and call any available chat
        prompt = self._build_prompt(question, full_ctx, language, intent)
        answer = self._call_any_llm(prompt)
        return answer, citations, "fallback"

    def _build_prompt(
        self, question: str, context: str, language: Optional[str], intent: str,
    ) -> str:
        lang = (language or "auto").lower()
        instr_lang = {
            "en": "Answer in clear English.",
            "fr": "Réponds en français clair.",
            "ar": "أجب باللغة العربية بوضوح.",
        }.get(lang, "Answer in the same language as the question.")
        if intent == "global":
            task = (
                "Synthesize the THEMATIC ANSWER using the community summaries "
                "and entities below. Quote entity names. Be concise and structured."
            )
        else:
            task = (
                "Answer the user's question using BOTH the entities/relations "
                "(knowledge graph) AND the transcript chunks (with timestamps). "
                "When you cite a chunk, include its [HH:MM:SS] timestamp. "
                "If the context does not contain the answer, say so."
            )
        return (
            f"You are a precise multilingual research assistant. {instr_lang}\n\n"
            f"TASK: {task}\n\n"
            f"CONTEXT:\n{context or '(no context found)'}\n\n"
            f"QUESTION: {question}\n\nANSWER:"
        )

    def _call_any_llm(self, prompt: str) -> str:
        if self.llm is None:
            return "(no LLM available)"
        for fn_name in ("chat_plain", "complete", "chat", "answer_question"):
            fn = getattr(self.llm, fn_name, None)
            if callable(fn):
                try:
                    out = fn(prompt=prompt, num_predict=self.cfg.llm_num_predict,
                             temperature=self.cfg.llm_temperature)
                    if isinstance(out, str):
                        return out
                    if isinstance(out, dict):
                        return out.get("text") or out.get("response") or ""
                    if hasattr(out, "answer"):
                        return out.answer
                except Exception:
                    continue
        return "(LLM call failed)"

    # ─────────────────────────────────────────────────────────────────
    # Formatting
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _format_chunks(hits: List[Any]) -> str:
        lines = ["## TRANSCRIPT EVIDENCE (with timestamps)"]
        for i, h in enumerate(hits, 1):
            text = (
                _attr(h, "text", "")
                or _attr(h, "content", "")
                or _attr(h, "search_text", "")
                or ""
            )
            text = (text or "").strip().replace("\n", " ")
            if len(text) > 320:
                text = text[:317] + "…"
            ts = _fmt_ts(_attr(h, "start_time", 0.0))
            speaker = _attr(h, "speaker", None) or "Unknown"
            vid = _attr(h, "video_id", "")
            lines.append(f"[{ts}] ({speaker}) [{vid}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _build_citations(
        vector_hits: List[Any], graph_res: Any,
    ) -> List[Dict[str, Any]]:
        citations: List[Dict[str, Any]] = []
        for h in vector_hits[:8]:
            citations.append({
                "chunk_id": _attr(h, "chunk_id", ""),
                "video_id": _attr(h, "video_id", ""),
                "start_time": float(_attr(h, "start_time", 0.0) or 0.0),
                "end_time": float(_attr(h, "end_time", 0.0) or 0.0),
                "start_fmt": _fmt_ts(_attr(h, "start_time", 0.0)),
                "speaker": _attr(h, "speaker", None),
                "language": _attr(h, "language", None),
                "score": float(_attr(h, "score", 0.0) or 0.0),
                "source": "vector",
            })
        if graph_res is not None:
            for n in (graph_res.nodes or [])[:6]:
                citations.append({
                    "entity_name": n.get("name"),
                    "entity_type": n.get("type"),
                    "is_seed": n.get("is_seed", False),
                    "source": "graph_entity",
                })
            for e in (graph_res.edges or [])[:6]:
                citations.append({
                    "edge": f"{e.get('src')} —[{e.get('type')}]→ {e.get('dst')}",
                    "source": "graph_relation",
                })
        return citations

    @staticmethod
    def _normalize_hit(h: Any) -> Dict[str, Any]:
        d = _to_dict(h)
        if "start_time" in d:
            d["start_fmt"] = _fmt_ts(d.get("start_time", 0.0))
        if "end_time" in d:
            d["end_fmt"] = _fmt_ts(d.get("end_time", 0.0))
        # trim text
        for k in ("text", "content", "search_text"):
            if k in d and isinstance(d[k], str) and len(d[k]) > 600:
                d[k] = d[k][:597] + "…"
        return d

    def _empty(self, t0: float, reason: str) -> Dict[str, Any]:
        return {
            "question": "",
            "mode_used": "none",
            "answer": "",
            "graph": None,
            "vector": [],
            "citations": [],
            "elapsed_seconds": round(time.time() - t0, 2),
            "error": reason,
        }