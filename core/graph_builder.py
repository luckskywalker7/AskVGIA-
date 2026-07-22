"""
core/graph_builder.py — End-to-end Knowledge Graph construction (v1.0)
========================================================================
Orchestrates the GraphRAG ingestion pipeline:

    chunks  ─► GraphExtractor ─► (entities, relations)
                                       │
                                       ▼
                              GraphStore (NetworkX)
                                       │
                                       ▼
                              detect_communities (Leiden)
                                       │
                                       ▼
                              LLM community summaries
                                       │
                                       ▼
                              persist (pickle + json)

Public API
----------
    builder = GraphBuilder(store=graph_store, extractor=graph_extractor,
                           llm_client=llm_client)

    res = builder.build_video_graph(
        video_id="abc123",
        chunks=semantic_chunks,
        run_communities=True,
        save=True,
    )
    # res = GraphBuildResult(n_nodes=42, n_edges=87, n_communities=5, ...)

    # incremental: when adding a new video to an existing graph
    res = builder.update_with_video(video_id="def456", chunks=new_chunks)

Author : Spiral father, 2026.
"""
from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

from core.graph_contracts import (
    GraphBuildResult,
    GraphEntity,
    GraphRelation,
)

logger = logging.getLogger(__name__)


def _cfg(name: str, default: Any) -> Any:
    try:
        from config.settings import settings  # type: ignore
        return getattr(settings, name, default)
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────
# GraphBuilder
# ─────────────────────────────────────────────────────────────────────
class GraphBuilder:
    """Pipeline driver: chunks → entities/relations → graph → communities."""

    def __init__(
        self,
        store: Any,                # core.graph_store.GraphStore
        extractor: Any,            # core.graph_extractor.GraphExtractor
        llm_client: Any = None,
        run_communities_default: bool = True,
        community_summarize_default: bool = True,
    ):
        self.store = store
        self.extractor = extractor
        self.llm = llm_client
        self.run_communities_default = bool(_cfg(
            "graph_run_communities", run_communities_default
        ))
        self.community_summarize_default = bool(_cfg(
            "graph_community_summarize", community_summarize_default
        ))
        logger.info(
            "GraphBuilder v1.0 | run_communities=%s | summarize=%s",
            self.run_communities_default,
            self.community_summarize_default,
        )

    # ─────────────────────────────────────────────────────────────────
    # Build for ONE video (most common entry point — called from /ingest)
    # ─────────────────────────────────────────────────────────────────
    def build_video_graph(
        self,
        video_id: str,
        chunks: List[Any],
        run_communities: Optional[bool] = None,
        summarize_communities: Optional[bool] = None,
        save: bool = True,
        replace_existing: bool = True,
    ) -> GraphBuildResult:
        """Extract from this video's chunks, merge into the global graph."""
        t0 = time.time()
        run_comm = self.run_communities_default if run_communities is None else run_communities
        do_sum = self.community_summarize_default if summarize_communities is None else summarize_communities

        if not chunks:
            logger.warning("GraphBuilder.build_video_graph(%s): no chunks", video_id)
            return GraphBuildResult(
                video_id=video_id, n_nodes=0, n_edges=0,
                n_communities=0, elapsed_seconds=time.time() - t0,
            )

        # Idempotence: drop the previous version of this video
        if replace_existing:
            try:
                self.store.delete_video(video_id)
            except Exception as exc:
                logger.warning("delete_video(%s) failed: %s", video_id, exc)

        # 1) Extract
        try:
            entities, relations = self.extractor.extract_from_chunks(chunks)
        except Exception as exc:
            logger.error("Extractor failed for %s: %s", video_id, exc)
            entities, relations = [], []

        # Force video_id stamping (defensive: extractor may forget)
        entities = self._stamp_video_id(entities, video_id)
        relations = self._stamp_video_id_rel(relations, video_id)

        # 2) Insert
        n_new_nodes = self.store.add_entities(entities)
        n_new_edges = self.store.add_relations(relations)

        # 3) Communities (over the WHOLE graph, not just this video)
        if run_comm:
            try:
                self.store.detect_communities(
                    llm_client=self.llm if do_sum else None,
                    summarize=do_sum,
                )
            except Exception as exc:
                logger.warning("detect_communities failed: %s", exc)

        # 4) Persist
        if save:
            try:
                self.store.save()
            except Exception as exc:
                logger.warning("GraphStore.save() failed: %s", exc)

        sample_ents = [e.name for e in entities[:8]]
        sample_rels = [
            f"{r.src} —{r.type}→ {r.dst}"
            for r in relations[:8]
        ]
        result = GraphBuildResult(
            video_id=video_id,
            n_nodes=self.store.graph.number_of_nodes(),
            n_edges=self.store.graph.number_of_edges(),
            n_communities=len(self.store.communities),
            elapsed_seconds=time.time() - t0,
            sample_entities=sample_ents,
            sample_relations=sample_rels,
        )
        logger.info(
            "GraphBuilder | video=%s | extracted %d ent / %d rel | "
            "graph_total: %d nodes / %d edges / %d comm | %.1fs",
            video_id, len(entities), len(relations),
            result.n_nodes, result.n_edges, result.n_communities,
            result.elapsed_seconds,
        )
        return result

    # ─────────────────────────────────────────────────────────────────
    # Convenience aliases
    # ─────────────────────────────────────────────────────────────────
    def update_with_video(
        self, video_id: str, chunks: List[Any], **kw: Any,
    ) -> GraphBuildResult:
        return self.build_video_graph(video_id=video_id, chunks=chunks, **kw)

    def rebuild_communities(
        self, summarize: bool = True, save: bool = True,
    ) -> int:
        try:
            comms = self.store.detect_communities(
                llm_client=self.llm if summarize else None,
                summarize=summarize,
            )
            if save:
                self.store.save()
            return len(comms)
        except Exception as exc:
            logger.warning("rebuild_communities failed: %s", exc)
            return 0

    # ─────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _stamp_video_id(
        entities: List[GraphEntity], video_id: str,
    ) -> List[GraphEntity]:
        for e in entities:
            e.video_ids.add(video_id)
            for m in e.mentions:
                if not m.video_id:
                    m.video_id = video_id
        return entities

    @staticmethod
    def _stamp_video_id_rel(
        relations: List[GraphRelation], video_id: str,
    ) -> List[GraphRelation]:
        for r in relations:
            for ev in r.evidence:
                if not ev.video_id:
                    ev.video_id = video_id
        return relations
