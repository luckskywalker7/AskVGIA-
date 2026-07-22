"""
core/graph_extractor.py — LLM-powered entity & relation extractor (v2.1)
=========================================================================

v2.1 fixes vs v2.0:
 PERF-1 extract_from_chunks() is now PARALLELIZED via ThreadPoolExecutor.
        Ollama handles multiple concurrent requests very well (internal
        request queue). On a laptop with a 4B model, 4 concurrent workers
        typically gives a 2.5x–3.5x speedup on 200-chunk videos.
        → 200 chunks × 45s sequential  ≈ 2h30
        → 200 chunks / 4 workers ≈ 40min  (realistic, not ideal linear)
        Workers are bounded by `max_workers` (default 4, settings-configurable
        via `graph_extractor_max_workers`). Set to 1 to restore v2.0 behavior.

 PERF-2 Order-preserving parallel execution: we keep the original chunk
        order in the returned lists so downstream code (GraphStore,
        deduplication by chunk_id) sees a stable, reproducible result.

 PERF-3 Thread-safe spaCy lazy loader: a module-level lock prevents
        multiple workers from loading the same spaCy model simultaneously
        (which would briefly spike memory and waste a few seconds).

 PERF-4 Thread-safe Ollama availability probe: wrapped in a small lock so
        only ONE worker ever performs the probe (others reuse the cached
        result).

v2.0 fixes preserved:
 FIX-1  Split extraction: TWO small LLM calls instead of one big call.
 FIX-2  Shorter, cleaner prompts.
 FIX-3  spaCy NER heuristic fallback (en/fr/ar) with co-occurrence.
 FIX-4  Adaptive timeout: skip relation call if entity call was slow.
 FIX-5  Text truncation reduced to 900 chars.
 FIX-6  Ollama availability cached per-session.

Public API (unchanged, backward-compatible):
    GraphExtractor(llm_client, ...).extract_from_chunk(chunk)
        -> (List[GraphEntity], List[GraphRelation])

    GraphExtractor(llm_client, ...).extract_from_chunks(chunks)
        -> (List[GraphEntity], List[GraphRelation])

Author : Spiral father, 2026.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from core.graph_contracts import (
    EntityMention,
    GraphEntity,
    GraphRelation,
    RelationEvidence,
    normalize_entity_type,
    normalize_relation_type,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Safe settings access
# ──────────────────────────────────────────────────────────────────────
try:
    from config.settings import settings as _settings
except Exception:
    _settings = None


def _cfg(name: str, default: Any) -> Any:
    if _settings is None:
        return default
    return getattr(_settings, name, default)


# ──────────────────────────────────────────────────────────────────────
# spaCy lazy loader  (FIX-3, PERF-3: thread-safe)
# ──────────────────────────────────────────────────────────────────────

_SPACY_NLP: Dict[str, Any] = {}   # lang_code -> nlp pipeline or None
_SPACY_LOCK = threading.Lock()    # PERF-3: protect concurrent model loads


def _get_spacy(lang: str) -> Optional[Any]:
    """
    Lazy-load the best available spaCy model for the requested language.
    Returns None (gracefully) if spaCy or the model is not installed.

    Thread-safe: multiple worker threads calling this concurrently will
    only trigger ONE model load per language (others wait on the lock).
    """
    key = lang[:2].lower()

    # Fast path (no lock) — CPython dict reads are atomic
    if key in _SPACY_NLP:
        return _SPACY_NLP[key]

    # Slow path (acquire lock, re-check, then load)
    with _SPACY_LOCK:
        if key in _SPACY_NLP:        # double-checked locking
            return _SPACY_NLP[key]

        try:
            import spacy  # type: ignore

            model_candidates = {
                "en": ["en_core_web_sm", "xx_ent_wiki_sm"],
                "fr": ["fr_core_news_sm", "xx_ent_wiki_sm"],
                "ar": ["xx_ent_wiki_sm"],
            }
            for model_name in model_candidates.get(key, ["xx_ent_wiki_sm"]):
                try:
                    nlp = spacy.load(
                        model_name, disable=["parser", "lemmatizer"]
                    )
                    _SPACY_NLP[key] = nlp
                    logger.info(
                        "spaCy model '%s' loaded for lang=%s", model_name, key
                    )
                    return nlp
                except OSError:
                    continue

            logger.debug(
                "No spaCy model found for lang=%s — regex fallback only", key
            )
            _SPACY_NLP[key] = None
            return None

        except ImportError:
            logger.debug("spaCy not installed — using regex fallback only")
            _SPACY_NLP[key] = None
            return None


# ──────────────────────────────────────────────────────────────────────
# Prompts  (FIX-2: shorter, split into entity + relation)
# ──────────────────────────────────────────────────────────────────────

# --- CALL 1: ENTITIES ONLY -------------------------------------------
ENTITY_PROMPT = """\
/nothink
Extract named entities from the transcript segment below.
Return a JSON object with key "entities" containing an array — no preamble, no markdown.

Allowed types:
  PERSON   — real named people (speakers, cited authors, historical figures)
  ORG      — organizations, companies, institutions, universities
  PLACE    — geographic locations, countries, cities, regions
  EVENT    — named events, conferences, historical events
  PRODUCT  — named products, software, tools, systems
  TECH     — technologies, protocols, algorithms, frameworks
  CONCEPT  — key abstract ideas, theories, methodologies discussed in depth
  WORK     — books, papers, datasets, standards
  DATE     — specific time references (years, periods)
  OTHER    — other significant named entities

Rules:
- Extract up to 15 entities. Prefer specificity over quantity.
- Use proper names, not generic words. No pronouns, fillers, or stop words.
- For CONCEPT: only concepts that are CENTRAL to this segment (not passing mentions).
- Each entity: {{"name": "...", "type": "...", "description": "<10 words max>"}}
- English labels even if the source text is Arabic or French.

[lang={lang}] [time={ts}]
{text}

JSON object with "entities" key:"""


# --- CALL 2: RELATIONS ONLY (given the entity list) ------------------
RELATION_PROMPT = """\
/nothink
Given these entities extracted from a transcript segment:
{entity_names}

Find meaningful, specific relations between them.
Return a JSON object with key "relations" containing an array — no preamble, no markdown.

Allowed relation types:
  discusses      — entity A is the topic of discussion by B
  mentions       — B briefly references A
  supports       — A provides evidence/backing for B
  opposes        — A contradicts or challenges B
  compares_to    — A is compared with B
  part_of        — A is a component/subset of B
  located_in     — A is physically situated inside B
  created_by     — A was created/invented by B
  works_at       — person A is affiliated with org B
  uses           — A employs/relies on B (tool, method, technique)
  causes         — A leads to / produces B
  precedes       — A comes before B in time or logic
  defined_as     — A is explained/defined as B
  related_to     — weak general link (use only if no better type fits)

Rules:
- Only emit relations where BOTH src and dst are in the entity list above.
- Each relation must be clearly supported by the segment text.
- Prefer specific types over 'related_to'.
- Each item: {{"src":"...", "dst":"...", "type":"...", "snippet":"<direct quote ≤6 words>"}}
- Max 12 relations.

Segment:
{text}

JSON object with "relations" key:"""


# ──────────────────────────────────────────────────────────────────────
# Config dataclass
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ExtractorConfig:
    model: Optional[str] = None
    max_text_chars: int = 1200          # increased for richer context
    temperature: float = 0.1
    entity_num_predict: int = 500       # more tokens for up to 15 entities
    relation_num_predict: int = 400     # more tokens for up to 12 relations
    entity_timeout_s: int = 90          # CPU: bumped from 45s
    relation_timeout_s: int = 90        # CPU: bumped from 60s
    slow_entity_threshold_s: float = 85.0  # must be < entity_timeout_s
    use_heuristic_fallback: bool = True
    min_entity_chars: int = 2
    max_entities_per_chunk: int = 15    # matches ENTITY_PROMPT cap
    max_relations_per_chunk: int = 12   # matches RELATION_PROMPT cap

    # PERF-1: parallel extraction
    # Ollama handles concurrent requests via its internal queue; on CPU
    # 3-4 workers usually saturates the model. Set to 1 to restore
    # fully sequential behavior (v2.0).
    max_workers: int = 1  # CPU-CRITICAL: sequential is faster than 4 starved workers


# ──────────────────────────────────────────────────────────────────────
# GraphExtractor
# ──────────────────────────────────────────────────────────────────────

class GraphExtractor:
    """
    Extracts entities & relations from semantic chunks.

    Strategy (per chunk):
      1. Try Ollama ENTITY call (fast, 45s timeout)
      2. If entity call OK and not too slow → try Ollama RELATION call (60s)
      3. If either LLM call fails/times out → spaCy NER + co-occurrence fallback
      4. If spaCy not installed → regex capitalized-word fallback

    Chunks are processed IN PARALLEL (PERF-1) via ThreadPoolExecutor.
    """

    def __init__(
        self,
        llm_client: Any = None,
        config: Optional[ExtractorConfig] = None,
    ):
        self.llm = llm_client
        self.cfg = config or ExtractorConfig()
        self.cfg.model = (
            self.cfg.model
            or _cfg("graph_extraction_model", None)
            or _cfg("ollama_chat_model", "qwen3.5:2b")
        )
        # PERF-1: allow max_workers to be overridden via settings
        try:
            settings_workers = _cfg("graph_extractor_max_workers", None)
            if settings_workers is not None:
                self.cfg.max_workers = max(1, int(settings_workers))
        except (TypeError, ValueError):
            pass

        self._ollama_url = _cfg(
            "ollama_base_url", "http://localhost:11434"
        ).rstrip("/")
        self._ollama_available: Optional[bool] = None   # FIX-6
        self._ollama_probe_lock = threading.Lock()      # PERF-4

        logger.info(
            "GraphExtractor v2.1 | model=%s | entity_timeout=%ds | "
            "relation_timeout=%ds | max_workers=%d | heuristic_fallback=%s",
            self.cfg.model,
            self.cfg.entity_timeout_s,
            self.cfg.relation_timeout_s,
            self.cfg.max_workers,
            self.cfg.use_heuristic_fallback,
        )

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def extract_from_chunk(
        self, chunk: Any,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        """Extract entities & relations from ONE SemanticChunk."""
        text = self._get_chunk_text(chunk)
        if not text or len(text.strip()) < 20:
            return [], []

        truncated_text = text[: self.cfg.max_text_chars]
        speaker = str(self._safe(chunk, "speaker", "unknown"))
        ts      = self._fmt_ts(self._safe(chunk, "start_time", 0.0))
        lang    = str(self._safe(chunk, "language", "en"))

        # ── Step 1: Extract entities ──────────────────────────────────
        entity_raw, entity_elapsed = self._call_entity(
            truncated_text, ts, lang
        )

        if not entity_raw:  # catches None AND empty list []
            if self.cfg.use_heuristic_fallback:
                logger.info("Entity LLM returned empty/None → heuristic fallback for chunk %s",
                            self._safe(chunk, "chunk_id", "?"))
                raw_full = self._heuristic_extract_full(text, lang)
                return self._build_entities_and_relations(chunk, raw_full)
            else:
                return [], []

        entity_names = [
            e.get("name", "").strip()
            for e in (entity_raw if isinstance(entity_raw, list) else [])
            if isinstance(e, dict) and e.get("name", "").strip()
        ]

        # ── Step 2: Extract relations (only if entities found & time OK)
        relation_raw: Optional[List[Any]] = None

        if (
            entity_names
            and entity_elapsed < self.cfg.slow_entity_threshold_s  # FIX-4
        ):
            relation_raw, _ = self._call_relation(
                truncated_text, entity_names
            )
        elif entity_names and entity_elapsed >= self.cfg.slow_entity_threshold_s:
            logger.info(
                "Entity call took %.1fs → skipping LLM relation call, "
                "using co-occurrence fallback",
                entity_elapsed,
            )
            relation_raw = self._cooccurrence_relations(entity_names)
        else:
            relation_raw = []

        # If relation LLM failed, use co-occurrence (FIX-3)
        if relation_raw is None:
            logger.debug("Relation LLM failed → co-occurrence fallback")
            relation_raw = self._cooccurrence_relations(entity_names)

        raw = {
            "entities": entity_raw if isinstance(entity_raw, list) else [],
            "relations": relation_raw if isinstance(relation_raw, list) else [],
        }
        return self._build_entities_and_relations(chunk, raw)

    def extract_from_chunks(
        self, chunks: List[Any],
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        """
        Extract from many chunks.

        PERF-1: executes `extract_from_chunk` in parallel via
        ThreadPoolExecutor. Results are gathered in ORIGINAL chunk order,
        so downstream merging stays deterministic.

        If cfg.max_workers <= 1, falls back to the v2.0 sequential path.
        """
        n = len(chunks)
        if n == 0:
            return [], []

        t0 = time.time()
        workers = max(1, int(self.cfg.max_workers or 1))

        # ── Sequential path (legacy, debugging, or single-worker) ─────
        if workers == 1:
            return self._extract_sequential(chunks, t0)

        # ── Parallel path ─────────────────────────────────────────────
        logger.info(
            "GraphExtractor PARALLEL start | chunks=%d | workers=%d",
            n, workers,
        )

        # per-index buckets — we preserve original order in the final lists
        ent_buckets: List[List[GraphEntity]] = [[] for _ in range(n)]
        rel_buckets: List[List[GraphRelation]] = [[] for _ in range(n)]
        completed = 0
        lock = threading.Lock()

        def _worker(idx: int, ch: Any) -> Tuple[int, List, List]:
            try:
                ents, rels = self.extract_from_chunk(ch)
                return idx, ents, rels
            except Exception as exc:
                logger.warning("Chunk %d extraction failed: %s", idx, exc)
                return idx, [], []

        with ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="graph-ext",
        ) as ex:
            futures = [
                ex.submit(_worker, i, ch) for i, ch in enumerate(chunks)
            ]
            for fut in as_completed(futures):
                try:
                    idx, ents, rels = fut.result()
                except Exception as exc:
                    logger.warning("Worker future failed: %s", exc)
                    continue
                ent_buckets[idx] = ents
                rel_buckets[idx] = rels

                with lock:
                    completed += 1
                    if completed % 5 == 0 or completed == n:
                        cum_e = sum(len(b) for b in ent_buckets)
                        cum_r = sum(len(b) for b in rel_buckets)
                        logger.info(
                            "GraphExtractor progress %d/%d "
                            "(cum: %d entities, %d relations)",
                            completed, n, cum_e, cum_r,
                        )

        # flatten in original chunk order
        all_entities: List[GraphEntity] = []
        all_relations: List[GraphRelation] = []
        for i in range(n):
            all_entities.extend(ent_buckets[i])
            all_relations.extend(rel_buckets[i])

        logger.info(
            "GraphExtractor done | %d chunks | %d entities | "
            "%d relations | %.1fs | workers=%d",
            n, len(all_entities), len(all_relations),
            time.time() - t0, workers,
        )
        return all_entities, all_relations

    # --- legacy sequential path (used when max_workers==1) ------------
    def _extract_sequential(
        self, chunks: List[Any], t0: float,
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        all_entities: List[GraphEntity] = []
        all_relations: List[GraphRelation] = []
        n = len(chunks)
        for i, ch in enumerate(chunks):
            try:
                ents, rels = self.extract_from_chunk(ch)
                all_entities.extend(ents)
                all_relations.extend(rels)
                if (i + 1) % 5 == 0 or (i + 1) == n:
                    logger.info(
                        "GraphExtractor progress %d/%d "
                        "(cum: %d entities, %d relations)",
                        i + 1, n,
                        len(all_entities), len(all_relations),
                    )
            except Exception as exc:
                logger.warning("Chunk %d extraction failed: %s", i, exc)

        logger.info(
            "GraphExtractor done | %d chunks | %d entities | "
            "%d relations | %.1fs | workers=1",
            n, len(all_entities), len(all_relations),
            time.time() - t0,
        )
        return all_entities, all_relations

    # ──────────────────────────────────────────────────────────────────
    # LLM calls  (FIX-1: two separate, smaller calls)
    # ──────────────────────────────────────────────────────────────────

    def _call_entity(
        self, text: str, ts: str, lang: str,
    ) -> Tuple[Optional[List[Any]], float]:
        """
        Call Ollama for entity extraction only.
        Returns (list_of_entity_dicts_or_None, elapsed_seconds).
        Thread-safe: requests.post is thread-safe; we open a fresh
        connection per call.
        """
        if not self._is_ollama_available():
            return None, 0.0

        prompt = ENTITY_PROMPT.format(text=text, ts=ts, lang=lang)
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.entity_num_predict,
                "top_p": 0.9,
                "num_ctx": 2048,
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json=payload,
                timeout=self.cfg.entity_timeout_s,
            )
            elapsed = time.time() - t0
            if resp.status_code != 200:
                logger.warning("Ollama entity call HTTP %d", resp.status_code)
                return None, elapsed
            raw_text = (resp.json().get("response") or "").strip()
            logger.info("Ollama entity raw_text (first 300): %s", raw_text[:300])
            parsed = self._safe_json_parse(raw_text)
            if isinstance(parsed, list):
                return parsed, elapsed
            if isinstance(parsed, dict):
                items = parsed.get("entities") or parsed.get("items") or []
                # FIX-2: LLM may return a single entity object instead of a
                # list (e.g. {"name": "X", "type": "Y"}).  Wrap it so the
                # caller always gets a list.
                if not items and parsed.get("name"):
                    items = [parsed]
                return items, elapsed
            logger.info("Entity LLM parse failed (raw: %.100s) → heuristic fallback will trigger", raw_text)
            return None, elapsed
        except requests.Timeout:
            elapsed = time.time() - t0
            logger.warning(
                "Ollama ENTITY call TIMEOUT after %.1fs (timeout=%ds)",
                elapsed, self.cfg.entity_timeout_s,
            )
            return None, elapsed
        except Exception as exc:
            elapsed = time.time() - t0
            logger.warning("Ollama entity call failed: %s", exc)
            return None, elapsed

    def _call_relation(
        self, text: str, entity_names: List[str],
    ) -> Tuple[Optional[List[Any]], float]:
        """
        Call Ollama for relation extraction only, given known entity names.
        Returns (list_of_relation_dicts_or_None, elapsed_seconds).
        """
        if not self._is_ollama_available():
            return None, 0.0

        names_str = "\n".join(f"- {n}" for n in entity_names[:10])
        prompt = RELATION_PROMPT.format(
            entity_names=names_str, text=text,
        )
        payload = {
            "model": self.cfg.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {
                "temperature": self.cfg.temperature,
                "num_predict": self.cfg.relation_num_predict,
                "top_p": 0.9,
                "num_ctx": 2048,
            },
        }

        t0 = time.time()
        try:
            resp = requests.post(
                f"{self._ollama_url}/api/generate",
                json=payload,
                timeout=self.cfg.relation_timeout_s,
            )
            elapsed = time.time() - t0
            if resp.status_code != 200:
                logger.warning(
                    "Ollama relation call HTTP %d", resp.status_code
                )
                return None, elapsed
            raw_text = (resp.json().get("response") or "").strip()
            logger.info("Ollama relation raw_text (first 300): %s", raw_text[:300])
            parsed = self._safe_json_parse(raw_text)
            if isinstance(parsed, list):
                logger.info("Relation parsed as list: %d items", len(parsed))
                return parsed, elapsed
            if isinstance(parsed, dict):
                items = parsed.get("relations") or parsed.get("items") or []
                logger.info("Relation parsed as dict, key 'relations': %d items", len(items))
                return items, elapsed
            logger.warning("Relation LLM parse failed (raw: %.100s)", raw_text)
            return None, elapsed
        except requests.Timeout:
            elapsed = time.time() - t0
            logger.warning(
                "Ollama RELATION call TIMEOUT after %.1fs (timeout=%ds)",
                elapsed, self.cfg.relation_timeout_s,
            )
            return None, elapsed
        except Exception as exc:
            elapsed = time.time() - t0
            logger.warning("Ollama relation call failed: %s", exc)
            return None, elapsed

    # ──────────────────────────────────────────────────────────────────
    # Ollama availability (FIX-6 + PERF-4: thread-safe cached probe)
    # ──────────────────────────────────────────────────────────────────

    def _is_ollama_available(self) -> bool:
        # Fast path — already probed
        if self._ollama_available is not None:
            return self._ollama_available

        with self._ollama_probe_lock:
            # Double-check inside the lock
            if self._ollama_available is not None:
                return self._ollama_available
            try:
                r = requests.get(
                    f"{self._ollama_url}/api/tags", timeout=3
                )
                self._ollama_available = r.status_code == 200
            except Exception:
                self._ollama_available = False
            if not self._ollama_available:
                logger.warning(
                    "Ollama NOT reachable at %s — heuristic mode only",
                    self._ollama_url,
                )
            return self._ollama_available

    # ──────────────────────────────────────────────────────────────────
    # JSON parse helper
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_json_parse(text: str) -> Optional[Any]:
        """
        Robust JSON parse:
          0. Strip Qwen3/3.5 <think>...</think> blocks first
          1. Direct json.loads
          2. Strip code fences then parse
          3. Extract first [...] or {...} block
        """
        # Strip thinking tags — qwen3.5:2b emits them even with think=False
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        text = re.sub(r"</?think>", "", text).strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        cleaned = re.sub(r"```(?:json)?", "", text).strip("` \n")
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        m = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        m = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
        return None

    # ──────────────────────────────────────────────────────────────────
    # Heuristic fallback — spaCy NER + co-occurrence  (FIX-3)
    # ──────────────────────────────────────────────────────────────────

    _SPACY_TYPE_MAP: Dict[str, str] = {
        "PERSON": "PERSON", "PER": "PERSON",
        "ORG": "ORG",
        "GPE": "PLACE", "LOC": "PLACE", "PLACE": "PLACE",
        "PRODUCT": "PRODUCT",
        "EVENT": "EVENT",
        "WORK_OF_ART": "WORK",
        "DATE": "DATE", "TIME": "DATE",
        "NORP": "ORG",
        "FACILITY": "PLACE",
        "LAW": "CONCEPT",
        "LANGUAGE": "CONCEPT",
        "MONEY": "OTHER", "PERCENT": "OTHER", "QUANTITY": "OTHER",
        "CARDINAL": "OTHER", "ORDINAL": "OTHER",
    }

    _PROPER_NOUN_RE = re.compile(
        r"\b([A-Z][a-zA-ZÀ-ÿ\u0600-\u06FF\.\-]{1,30}"
        r"(?:\s+[A-Z][a-zA-ZÀ-ÿ\u0600-\u06FF\.\-]{1,30}){0,3})\b"
    )
    _STOPWORDS = {
        # English
        "I", "He", "She", "It", "We", "They", "You", "The", "This",
        "That", "These", "Those", "And", "But", "Or", "So", "As",
        "Mr", "Mrs", "Dr", "In", "On", "At", "By", "For", "Of",
        "To", "Do", "Be", "Is", "Are", "Was", "Were", "Has", "Have",
        "Had", "Can", "Could", "Will", "Would", "May", "Might",
        "Not", "No", "Yes", "My", "Our", "Your", "His", "Her", "Its",
        "Their", "Who", "What", "When", "Where", "Why", "How",
        # French — common words that start with a capital at sentence-start
        "Le", "La", "Les", "Un", "Une", "Des", "Du", "Au", "Aux",
        "En", "Et", "Ou", "Ni", "Mais", "Donc", "Or", "Car",
        "Ce", "Cet", "Cette", "Ces", "Mon", "Ma", "Mes", "Ton", "Ta",
        "Tes", "Son", "Sa", "Ses", "Notre", "Votre", "Leur", "Leurs",
        "Qui", "Que", "Quoi", "Quand", "Comment", "Pourquoi",
        "Avoir", "Être", "Faire", "Dire", "Aller", "Voir", "Vouloir",
        "Alors", "Aussi", "Bien", "Mais", "Même", "Plus", "Très",
        "Bonjour", "Bonsoir", "Merci", "Voici", "Voilà",
        "Aujourd", "Souviens", "Imagine", "Imaginons",
        "Je", "Tu", "Il", "Elle", "Nous", "Vous", "Ils", "Elles",
        "Me", "Te", "Se", "Lui", "Eux",
        # Arabic — common words (transliterated capitalised form never applies,
        # but included for safety when mixing scripts is attempted)
        "هذا", "هذه", "ذلك", "تلك", "هو", "هي", "نحن", "أنت", "أنتم",
        "في", "من", "إلى", "على", "مع", "عن", "بعد", "قبل",
        "و", "أو", "لكن", "لأن", "إذا", "عندما", "حيث",
    }

    def _heuristic_extract_full(
        self, text: str, lang: str = "en",
    ) -> Dict[str, Any]:
        """
        Full heuristic: entities via spaCy (or regex), relations via
        co-occurrence. Always produces edges so the graph is never empty.
        """
        entities = self._spacy_or_regex_entities(text, lang)
        names = [e["name"] for e in entities]
        relations = self._cooccurrence_relations(names)
        return {"entities": entities, "relations": relations}

    def _spacy_or_regex_entities(
        self, text: str, lang: str,
    ) -> List[Dict[str, Any]]:
        """Try spaCy first, fall back to regex."""
        nlp = _get_spacy(lang)
        if nlp is not None:
            try:
                doc = nlp(text[:1000])
                seen: Dict[str, str] = {}
                for ent in doc.ents:
                    name = ent.text.strip()
                    if len(name) < 2:
                        continue
                    if name in self._STOPWORDS:
                        continue
                    canonical_type = self._SPACY_TYPE_MAP.get(
                        ent.label_, "OTHER"
                    )
                    if name not in seen:
                        seen[name] = canonical_type
                result = [
                    {"name": n, "type": t, "description": "", "raw_surface": n}
                    for n, t in seen.items()
                ][: self.cfg.max_entities_per_chunk]
                if result:
                    return result
            except Exception as exc:
                logger.debug("spaCy NER failed: %s", exc)

        # Regex fallback
        candidates: Dict[str, int] = {}
        for m in self._PROPER_NOUN_RE.finditer(text):
            name = m.group(1).strip(" .,;:!?")
            if name in self._STOPWORDS or len(name) < self.cfg.min_entity_chars:
                continue
            candidates[name] = candidates.get(name, 0) + 1
        ranked = sorted(candidates.items(), key=lambda x: -x[1])
        return [
            {"name": n, "type": "OTHER", "description": "", "raw_surface": n}
            for n, _ in ranked[: self.cfg.max_entities_per_chunk]
        ]

    @staticmethod
    def _cooccurrence_relations(
        names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Build co-occurrence edges: every pair of entities that appears in
        the same chunk gets a 'related_to' relation. Capped at 8 edges.
        """
        relations: List[Dict[str, Any]] = []
        limit = 8
        for i, src in enumerate(names):
            for dst in names[i + 1:]:
                if len(relations) >= limit:
                    return relations
                relations.append({
                    "src": src,
                    "dst": dst,
                    "type": "related_to",
                    "snippet": "co-occurrence in same segment",
                })
        return relations

    # ──────────────────────────────────────────────────────────────────
    # Build canonical GraphEntity / GraphRelation objects
    # ──────────────────────────────────────────────────────────────────

    def _build_entities_and_relations(
        self, chunk: Any, raw: Dict[str, Any],
    ) -> Tuple[List[GraphEntity], List[GraphRelation]]:
        if not isinstance(raw, dict):
            return [], []

        chunk_id = str(self._safe(chunk, "chunk_id", ""))
        video_id = str(self._safe(chunk, "video_id", ""))
        ts       = float(self._safe(chunk, "start_time", 0.0))
        end      = self._safe(chunk, "end_time", None)
        speaker  = self._safe(chunk, "speaker", None)
        language = self._safe(chunk, "language", None)
        chunk_text = self._get_chunk_text(chunk)

        ent_list = (raw.get("entities") or [])[: self.cfg.max_entities_per_chunk]
        rel_list = (raw.get("relations") or [])[: self.cfg.max_relations_per_chunk]

        entities: List[GraphEntity] = []
        names_seen: set = set()

        for e in ent_list:
            if not isinstance(e, dict):
                continue
            name = str(e.get("name", "")).strip()
            if len(name) < self.cfg.min_entity_chars:
                continue
            if name in self._STOPWORDS or name in names_seen:
                continue
            names_seen.add(name)

            mention = EntityMention(
                video_id=video_id,
                chunk_id=chunk_id,
                timestamp=ts,
                end_time=float(end) if end is not None else None,
                speaker=speaker,
                language=language,
                raw_surface=str(e.get("raw_surface", name))[:200],
                confidence=0.9,
            )
            ent = GraphEntity(
                name=name,
                type=normalize_entity_type(e.get("type", "OTHER")),
                description=str(e.get("description", ""))[:300],
                aliases={str(e.get("raw_surface", name))}
                    if e.get("raw_surface") else set(),
            )
            ent.add_mention(mention)
            entities.append(ent)

        relations: List[GraphRelation] = []
        valid_names = {ent.name for ent in entities}

        for r in rel_list:
            if not isinstance(r, dict):
                continue
            src = str(r.get("src", "")).strip()
            dst = str(r.get("dst", "")).strip()
            if not src or not dst or src == dst:
                continue
            if src not in valid_names or dst not in valid_names:
                if (
                    src.lower() not in chunk_text.lower()
                    or dst.lower() not in chunk_text.lower()
                ):
                    snippet = str(r.get("snippet", ""))
                    if snippet != "co-occurrence in same segment":
                        continue

            ev = RelationEvidence(
                video_id=video_id,
                chunk_id=chunk_id,
                timestamp=ts,
                end_time=float(end) if end is not None else None,
                speaker=speaker,
                snippet=str(r.get("snippet", ""))[:240],
                confidence=0.85,
            )
            rel = GraphRelation(
                src=src,
                dst=dst,
                type=normalize_relation_type(r.get("type", "related_to")),
                description=str(r.get("snippet", ""))[:200],
            )
            rel.add_evidence(ev)
            relations.append(rel)

        return entities, relations

    # ──────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_chunk_text(chunk: Any) -> str:
        for attr in ("text", "original_text", "normalized_text", "search_text"):
            v = GraphExtractor._safe(chunk, attr, None)
            if v:
                return str(v)
        return ""

    @staticmethod
    def _safe(obj: Any, name: str, default: Any) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    @staticmethod
    def _fmt_ts(seconds: Any) -> str:
        try:
            s = float(seconds)
        except (TypeError, ValueError):
            return "00:00"
        m, s = divmod(int(s), 60)
        return f"{m:02d}:{s:02d}"