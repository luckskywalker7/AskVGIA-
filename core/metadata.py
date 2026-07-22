"""
core/metadata.py — Video-level Metadata Registry v2.0
=======================================================

NOUVEAUTÉS v2.0 (vs v1.0)
--------------------------

META-1   domain + sub_domain intégrés dans VideoMetadata
    Chaque vidéo est catégorisée selon la taxonomie DOMAIN_TAXONOMY
    (Technology, Science, Education, Business, Health, History, Culture,
    Sports, News, Entertainment, Religion, Law, Language, Other).

META-2   full_text : la transcription complète est stockée dans
    VideoMetadata.extra["full_text"] pour permettre la recherche
    plein-texte et l'affichage du script complet.

META-3   chunk_topics : la liste des sujets par chunk est stockée
    dans VideoMetadata.extra["chunk_topics"] pour navigation fine.

META-4   integration VideoSummary : si la chunking result contient
    un VideoSummary (chunking.py v4.0), il est utilisé directement
    au lieu de relancer un appel LLM séparé.

META-5   VideoMetadataRegistry.search_by_domain(domain) : chercher
    toutes les vidéos d'un domaine donné.

META-6   VideoMetadataRegistry.list_domains() : liste les domaines
    disponibles dans la base.

Public API:
    registry.get(video_id)                          -> Optional[VideoMetadata]
    registry.list_all()                             -> List[VideoMetadata]
    registry.search(query)                          -> List[VideoMetadata]
    registry.search_by_domain(domain)               -> List[VideoMetadata]
    registry.list_domains()                         -> Dict[str, int]
    registry.delete(video_id)
    registry.generate_and_register(...)             -> VideoMetadata
    registry.register_from_video_summary(...)       -> VideoMetadata
    registry.update_title(video_id, title)
    registry.save() / registry.load()
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# VideoMetadata dataclass
# ─────────────────────────────────────────────────────────────────────

@dataclass
class VideoMetadata:
    """
    Video-level record stored alongside chunk-level FAISS data.

    v2.0 additions:
        domain          : primary category (from DOMAIN_TAXONOMY)
        sub_domain      : secondary category (optional)
        full_text       : complete concatenated transcript (in extra)
        chunk_topics    : per-chunk topic labels (in extra)
    """
    video_id: str
    title: str = ""
    summary: str = ""
    keywords: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    language: str = "unknown"
    duration_seconds: Optional[float] = None
    total_chunks: int = 0
    has_visual: bool = False
    file_path: Optional[str] = None
    # v2.0
    domain: str = "Other"
    sub_domain: str = ""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    extra: Dict[str, Any] = field(default_factory=dict)

    # ── helpers ──────────────────────────────────────────────────────

    def touch(self) -> None:
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "VideoMetadata":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def matches_query(self, query: str) -> bool:
        """
        Keyword-based relevance check for discovery.
        Searches: title, summary, keywords, topics, domain.
        Also searches full_text snippet if available.
        """
        q = query.lower()
        q_words = [w for w in q.split() if len(w) > 2]
        haystack = " ".join([
            self.title.lower(),
            self.summary.lower(),
            " ".join(self.keywords).lower(),
            " ".join(self.topics).lower(),
            self.domain.lower(),
            self.sub_domain.lower(),
        ])
        # Quick check on main fields
        if any(word in haystack for word in q_words):
            return True
        # Check full_text snippet (first 500 chars)
        full_text = str(self.extra.get("full_text", ""))[:500].lower()
        if full_text and any(word in full_text for word in q_words):
            return True
        return False

    @property
    def full_text(self) -> str:
        """Complete transcript text stored in extra."""
        return str(self.extra.get("full_text", ""))

    @property
    def chunk_topics(self) -> List[str]:
        """Per-chunk topic list stored in extra."""
        return list(self.extra.get("chunk_topics", []))


# ─────────────────────────────────────────────────────────────────────
# LLM summarization helpers
# ─────────────────────────────────────────────────────────────────────

_SUMMARY_PROMPT = """\
You are a video content analyst. Below are representative text excerpts from a video.
Produce a JSON object with the following fields and nothing else:
{{
  "title": "short descriptive title for the video (max 10 words)",
  "summary": "3-5 sentence overview of what this video covers",
  "keywords": ["8 to 15 specific keywords or short phrases"],
  "topics": ["3 to 8 specific subjects discussed in this video"],
  "domain": "ONE category from: Technology, Science, Education, Business, Health, History, Culture, Sports, News, Entertainment, Religion, Law, Language, Other",
  "sub_domain": "optional secondary category from the same list, or empty string",
  "language": "dominant language code: en | ar | fr | es | de | it | pt | unknown"
}}

Video excerpts (first {n_excerpts} chunks):
---
{excerpts}
---
Respond ONLY with the JSON object. No markdown fences, no preamble.\
"""


def _build_excerpts(chunks: List[Any], max_chars: int = 2000) -> str:
    parts: List[str] = []
    total = 0
    for chunk in chunks:
        text = ""
        if isinstance(chunk, dict):
            text = (
                chunk.get("text")
                or chunk.get("normalized_text")
                or chunk.get("original_text") or ""
            )
        else:
            text = (
                getattr(chunk, "text", "")
                or getattr(chunk, "normalized_text", "")
                or getattr(chunk, "original_text", "")
            )
        text = str(text).strip()
        if not text or len(text) < 10:
            continue
        snippet = text[:200].replace("\n", " ").strip()
        parts.append(f"• {snippet}")
        total += len(snippet)
        if total >= max_chars:
            break
    return "\n".join(parts) or "(no text available)"


def _call_llm_for_summary(
    prompt: str,
    llm_client: Any,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3.5:2b",
) -> Optional[Dict[str, Any]]:
    raw: Optional[str] = None

    # Try llm_client
    if llm_client is not None:
        for method_name in ["raw_complete", "complete", "generate", "chat"]:
            fn = getattr(llm_client, method_name, None)
            if callable(fn):
                try:
                    result = fn(
                        prompt=prompt,
                        max_tokens=400,
                        temperature=0.1,
                        json_mode=True,
                    )
                    if isinstance(result, str):
                        raw = result
                    elif isinstance(result, dict):
                        raw = (
                            result.get("text")
                            or result.get("content")
                            or result.get("response") or ""
                        )
                    break
                except Exception as e:
                    logger.debug("llm_client.%s failed: %s", method_name, e)

    # Try chat_json from llm_client module
    if not raw:
        try:
            from core.llm_client import chat_json  # type: ignore
            parsed = chat_json(
                prompt,
                system="You are a precise video analyst. Output only JSON.",
                num_predict=400,
                temperature=0.1,
            )
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception as e:
            logger.debug("chat_json failed: %s", e)

    # Direct Ollama fallback
    if not raw:
        try:
            payload = json.dumps({
                "model": model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 400},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{ollama_url}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                raw = body.get("response", "")
        except Exception as e:
            logger.debug("Direct Ollama call failed: %s", e)

    if not raw:
        return None

    # Parse JSON
    raw = raw.strip()
    # Strip markdown fences
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


# ─────────────────────────────────────────────────────────────────────
# Heuristic helpers
# ─────────────────────────────────────────────────────────────────────

def _heuristic_summary(chunks: List[Any], max_chars: int = 500) -> str:
    parts: List[str] = []
    total = 0
    for chunk in chunks[:5]:
        text = ""
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("original_text") or ""
        else:
            text = getattr(chunk, "text", "") or getattr(chunk, "original_text", "")
        text = str(text).strip()
        if text:
            parts.append(text[:200])
            total += len(text)
        if total >= max_chars:
            break
    return " ".join(parts)[:max_chars] or "No summary available."


def _heuristic_keywords(chunks: List[Any], top_n: int = 12) -> List[str]:
    import re
    from collections import Counter

    STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "on",
        "at", "by", "for", "with", "as", "and", "or", "but", "not", "this",
        "that", "it", "its", "i", "we", "you", "he", "she", "they", "so",
        "if", "then", "from", "into", "up", "about", "there", "what", "when",
        "how", "which", "who", "all", "just", "also", "more", "than", "our",
        "very", "no", "my", "your", "his", "her", "their",
        "le", "la", "les", "des", "un", "une", "est", "dans", "pour", "avec",
        "في", "من", "على", "إلى", "مع", "هذا", "هذه", "كان", "التي", "الذي",
    }

    counter: Counter = Counter()
    for chunk in chunks:
        if isinstance(chunk, dict):
            text = chunk.get("text") or chunk.get("normalized_text") or ""
        else:
            text = getattr(chunk, "text", "") or getattr(chunk, "normalized_text", "")
        words = re.findall(r"\b[a-zA-Z\u0600-\u06FF]{4,}\b", str(text).lower())
        for w in words:
            if w not in STOP:
                counter[w] += 1

    return [word for word, _ in counter.most_common(top_n)]


# ─────────────────────────────────────────────────────────────────────
# VideoMetadataRegistry
# ─────────────────────────────────────────────────────────────────────

class VideoMetadataRegistry:
    """
    Persistent JSON registry of VideoMetadata records.

    v2.0 additions:
        - domain / sub_domain stored per video
        - full_text in extra
        - search_by_domain() for category browsing
        - list_domains() for discovery UI
        - register_from_video_summary() for seamless integration
          with chunking.py v4.0 VideoSummary
    """

    _FILENAME = "video_metadata.json"

    def __init__(self, persist_dir: str = "data/faiss_store"):
        self._dir = Path(persist_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / self._FILENAME
        self._records: Dict[str, VideoMetadata] = {}
        self.load()

    # ── Persistence ──────────────────────────────────────────────────

    def load(self) -> None:
        if not self._path.exists():
            logger.debug("VideoMetadataRegistry: no file at %s, starting empty", self._path)
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data.get("records", []):
                try:
                    meta = VideoMetadata.from_dict(entry)
                    self._records[meta.video_id] = meta
                except Exception as e:
                    logger.warning("Failed to load metadata record: %s", e)
            logger.info(
                "VideoMetadataRegistry: loaded %d records from %s",
                len(self._records), self._path,
            )
        except Exception as e:
            logger.error("VideoMetadataRegistry load failed: %s", e)

    def save(self) -> None:
        try:
            data = {
                "version": "2.0",
                "saved_at": time.time(),
                "count": len(self._records),
                "records": [m.to_dict() for m in self._records.values()],
            }
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(
                "VideoMetadataRegistry: saved %d records to %s",
                len(self._records), self._path,
            )
        except Exception as e:
            logger.error("VideoMetadataRegistry save failed: %s", e)

    # ── CRUD ─────────────────────────────────────────────────────────

    def register(self, meta: VideoMetadata) -> VideoMetadata:
        meta.touch()
        self._records[meta.video_id] = meta
        self.save()
        return meta

    def get(self, video_id: str) -> Optional[VideoMetadata]:
        return self._records.get(video_id)

    def delete(self, video_id: str) -> bool:
        if video_id in self._records:
            del self._records[video_id]
            self.save()
            return True
        return False

    def update_title(self, video_id: str, title: str) -> bool:
        if video_id in self._records:
            self._records[video_id].title = title
            self._records[video_id].touch()
            self.save()
            return True
        return False

    def list_all(self) -> List[VideoMetadata]:
        return sorted(
            self._records.values(),
            key=lambda m: m.created_at,
            reverse=True,
        )

    # ── Search ───────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        limit: int = 20,
        domain_filter: Optional[str] = None,
    ) -> List[VideoMetadata]:
        """
        Keyword search across title, summary, keywords, topics, domain.
        Optional domain_filter restricts to one category.
        """
        q_words = [w for w in query.lower().split() if len(w) > 2]
        if not q_words:
            results = self.list_all()
            if domain_filter:
                results = [r for r in results if r.domain == domain_filter]
            return results[:limit]

        scored: List[tuple] = []
        for meta in self._records.values():
            if domain_filter and meta.domain != domain_filter:
                continue
            haystack = " ".join([
                meta.title.lower(),
                meta.summary.lower(),
                " ".join(meta.keywords).lower(),
                " ".join(meta.topics).lower(),
                meta.domain.lower(),
                meta.sub_domain.lower(),
            ])
            score = sum(1 for w in q_words if w in haystack)
            if score > 0:
                scored.append((score, meta))

        scored.sort(key=lambda x: -x[0])
        return [m for _, m in scored[:limit]]

    def search_by_domain(
        self,
        domain: str,
        limit: int = 50,
    ) -> List[VideoMetadata]:
        """
        Return all videos categorized under a given domain.
        Case-insensitive match.
        """
        domain_lower = domain.lower()
        results = [
            m for m in self._records.values()
            if m.domain.lower() == domain_lower
            or m.sub_domain.lower() == domain_lower
        ]
        return sorted(results, key=lambda m: m.created_at, reverse=True)[:limit]

    def list_domains(self) -> Dict[str, int]:
        """
        Return a dict mapping domain name → number of videos.
        Useful for building a category browser UI.
        """
        counts: Dict[str, int] = {}
        for meta in self._records.values():
            d = meta.domain or "Other"
            counts[d] = counts.get(d, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: -x[1]))

    # ── Integration with chunking.py v4.0 VideoSummary ───────────────

    def register_from_video_summary(
        self,
        video_summary: Any,   # core.chunking.VideoSummary
        title: str = "",
        file_path: Optional[str] = None,
        has_visual: bool = False,
        duration_seconds: Optional[float] = None,
    ) -> VideoMetadata:
        """
        Build and register a VideoMetadata directly from a VideoSummary
        produced by chunking.py v4.0.

        This avoids a redundant LLM call — the summary has already been
        generated during chunking.
        """
        # Handle both VideoSummary objects and plain dicts
        def _get(obj: Any, key: str, default: Any = "") -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        video_id = _get(video_summary, "video_id", "unknown")
        full_text = str(_get(video_summary, "full_text", ""))
        summary = str(_get(video_summary, "summary", ""))
        topics = list(_get(video_summary, "topics", []))
        keywords = list(_get(video_summary, "keywords", []))
        domain = str(_get(video_summary, "domain", "Other"))
        sub_domain = str(_get(video_summary, "sub_domain", ""))
        language = str(_get(video_summary, "language", "unknown"))
        total_chunks = int(_get(video_summary, "total_chunks", 0))
        chunk_topics = list(_get(video_summary, "chunk_topics", []))
        dur = float(_get(video_summary, "duration_seconds", 0.0))

        if not title and file_path:
            title = Path(file_path).stem.replace("_", " ").replace("-", " ").strip()
        if not title:
            title = video_id

        meta = VideoMetadata(
            video_id=video_id,
            title=title,
            summary=summary,
            keywords=keywords[:15],
            topics=topics[:8],
            language=language,
            duration_seconds=duration_seconds or (dur if dur > 0 else None),
            total_chunks=total_chunks,
            has_visual=has_visual,
            file_path=file_path,
            domain=domain,
            sub_domain=sub_domain,
            extra={
                "full_text": full_text,
                "chunk_topics": chunk_topics,
            },
        )

        logger.info(
            "VideoMetadata registered from VideoSummary: "
            "video_id=%s domain=%s topics=%s keywords=%s",
            video_id, domain, topics[:3], keywords[:5],
        )

        return self.register(meta)

    # ── LLM-powered registration (kept from v1.0) ─────────────────────

    def generate_and_register(
        self,
        video_id: str,
        chunks: List[Any],
        title: str = "",
        language: str = "unknown",
        duration_seconds: Optional[float] = None,
        total_chunks: int = 0,
        has_visual: bool = False,
        file_path: Optional[str] = None,
        llm_client: Any = None,
        ollama_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3.5:2b",
        video_summary: Any = None,       # v2.0: accept VideoSummary directly
    ) -> VideoMetadata:
        """
        Generate a VideoMetadata record for a newly-ingested video.

        v2.0: If a VideoSummary is provided (from chunking.py v4.0),
        it is used directly — no redundant LLM call.

        Otherwise falls back to LLM-based extraction, then heuristics.
        """
        # ── Fast path: use VideoSummary from chunking ────────────────
        if video_summary is not None:
            return self.register_from_video_summary(
                video_summary=video_summary,
                title=title,
                file_path=file_path,
                has_visual=has_visual,
                duration_seconds=duration_seconds,
            )

        # ── Fallback title ───────────────────────────────────────────
        if not title and file_path:
            title = Path(file_path).stem.replace("_", " ").replace("-", " ").strip()
        if not title:
            title = video_id

        n_chunks = total_chunks or (len(chunks) if chunks else 0)

        if not chunks:
            meta = VideoMetadata(
                video_id=video_id,
                title=title,
                summary="No content available — transcription may have failed.",
                keywords=[],
                topics=[],
                language=language,
                duration_seconds=duration_seconds,
                total_chunks=n_chunks,
                has_visual=has_visual,
                file_path=file_path,
                domain="Other",
                sub_domain="",
            )
            return self.register(meta)

        # ── Build full transcript text ────────────────────────────────
        text_parts: List[str] = []
        for chunk in chunks:
            t = ""
            if isinstance(chunk, dict):
                t = chunk.get("text") or chunk.get("original_text") or ""
            else:
                t = getattr(chunk, "text", "") or getattr(chunk, "original_text", "")
            t = str(t).strip()
            if t:
                text_parts.append(t)
        full_text = " ".join(text_parts)

        # ── LLM call ─────────────────────────────────────────────────
        excerpts = _build_excerpts(chunks, max_chars=2000)
        n_excerpts = min(len(chunks), 15)
        prompt = _SUMMARY_PROMPT.format(
            n_excerpts=n_excerpts,
            excerpts=excerpts,
        )

        parsed: Optional[Dict[str, Any]] = None
        try:
            parsed = _call_llm_for_summary(
                prompt=prompt,
                llm_client=llm_client,
                ollama_url=ollama_url,
                model=ollama_model,
            )
        except Exception as e:
            logger.warning("generate_and_register: LLM call failed: %s", e)

        # ── Build VideoMetadata ──────────────────────────────────────
        if parsed:
            domain = str(parsed.get("domain", "Other") or "Other").strip()
            # Import DOMAIN_TAXONOMY lazily
            try:
                from core.chunking import DOMAIN_TAXONOMY  # type: ignore
                if domain not in DOMAIN_TAXONOMY:
                    domain = "Other"
            except ImportError:
                pass

            sub_domain = str(parsed.get("sub_domain", "") or "").strip()

            meta = VideoMetadata(
                video_id=video_id,
                title=str(parsed.get("title", "") or "").strip() or title,
                summary=str(parsed.get("summary", "") or "").strip(),
                keywords=[str(k) for k in parsed.get("keywords", []) if k][:15],
                topics=[str(t) for t in parsed.get("topics", []) if t][:8],
                language=str(parsed.get("language", language) or language),
                duration_seconds=duration_seconds,
                total_chunks=n_chunks,
                has_visual=has_visual,
                file_path=file_path,
                domain=domain,
                sub_domain=sub_domain,
                extra={"full_text": full_text},
            )
            logger.info(
                "VideoMetadata generated: video_id=%s domain=%s topics=%s",
                video_id, meta.domain, meta.topics,
            )
        else:
            logger.warning("LLM unavailable for %s — using heuristics", video_id)
            keywords = _heuristic_keywords(chunks)

            # Domain detection heuristic
            try:
                from core.chunking import detect_domain_heuristic  # type: ignore
                domain, sub_domain = detect_domain_heuristic(full_text[:2000], keywords)
            except ImportError:
                domain, sub_domain = "Other", ""

            meta = VideoMetadata(
                video_id=video_id,
                title=title,
                summary=_heuristic_summary(chunks),
                keywords=keywords,
                topics=[],
                language=language,
                duration_seconds=duration_seconds,
                total_chunks=n_chunks,
                has_visual=has_visual,
                file_path=file_path,
                domain=domain,
                sub_domain=sub_domain,
                extra={"full_text": full_text},
            )

        return self.register(meta)

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, video_id: str) -> bool:
        return video_id in self._records


# ─────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────

_registry_instance: Optional[VideoMetadataRegistry] = None


def get_registry(persist_dir: str = "data/faiss_store") -> VideoMetadataRegistry:
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = VideoMetadataRegistry(persist_dir=persist_dir)
    return _registry_instance


__all__ = [
    "VideoMetadata",
    "VideoMetadataRegistry",
    "get_registry",
]