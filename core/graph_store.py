"""
core/graph_store.py — NetworkX-backed Knowledge Graph store (v1.0)
====================================================================
Persistence + community detection + local/global search primitives
for the Agentic GraphRAG pipeline.

Design principles
-----------------
1. **Single canonical store** : one MultiDiGraph holds ALL videos.
   Per-video subgraphs are obtained via `subgraph_for_video()`.
2. **Entity merging** : same canonical name (case-insensitive) → same node.
   Aliases are stored on the node and used for fuzzy matching at query time.
3. **Persistence** : pickle (fast, full fidelity) + JSON (debuggable, portable).
4. **Community detection** : Leiden via `python-igraph + leidenalg` if
   available, else a robust greedy fallback (NetworkX greedy_modularity).
5. **Community summarization** : LLM-generated 2-4 sentence reports keyed by
   community_id. Used by GLOBAL search (Microsoft GraphRAG style).
6. **Local search** : seed entities → 1-2 hop neighborhood → return nodes,
   edges, supporting chunk_ids (for the existing FAISS+BM25 retriever).

Public API
----------
    store = GraphStore(persist_dir="data/graph_store")
    store.add_entities(entities)
    store.add_relations(relations)
    store.detect_communities(llm_client=...)        # builds + summarizes
    store.save() / store.load()
    store.delete_video(video_id)

    res = store.local_search(seed_entities=["BB84"], hops=1, max_nodes=30)
    res = store.global_search(query="main themes", top_k=3)

Author : Spiral father, 2026.
"""
from __future__ import annotations

import json
import logging
import pickle
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx

from core.graph_contracts import (
    EntityMention,
    GraphCommunity,
    GraphEntity,
    GraphRelation,
    GraphSearchResult,
    RelationEvidence,
    normalize_entity_type,
    normalize_relation_type,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Optional deps
# ─────────────────────────────────────────────────────────────────────
try:
    import igraph as ig
    import leidenalg as la
    _LEIDEN_AVAILABLE = True
except Exception:  # pragma: no cover
    ig = None
    la = None
    _LEIDEN_AVAILABLE = False

try:
    from rapidfuzz import fuzz, process as _fuzzproc
    _FUZZY_AVAILABLE = True
except Exception:  # pragma: no cover
    fuzz = None
    _fuzzproc = None
    _FUZZY_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
# Safe settings access
# ─────────────────────────────────────────────────────────────────────
def _cfg(name: str, default: Any) -> Any:
    try:
        from config.settings import settings  # type: ignore
        return getattr(settings, name, default)
    except Exception:
        return default


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────
def _canon_key(name: str) -> str:
    """Canonical lookup key for an entity name (case + whitespace + diacritics-tolerant)."""
    if not name:
        return ""
    s = re.sub(r"\s+", " ", str(name).strip().lower())
    s = re.sub(r"[^\w\s\u0600-\u06FF\u00C0-\u017F]+", "", s)
    return s


def _short(text: str, n: int = 280) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


# ─────────────────────────────────────────────────────────────────────
# GraphStore
# ─────────────────────────────────────────────────────────────────────
class GraphStore:
    """NetworkX-backed canonical knowledge graph + community layer."""

    PICKLE_NAME = "graph.pkl"
    JSON_NAME = "graph.json"
    COMM_NAME = "communities.json"

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = Path(
            persist_dir or _cfg("graph_store_dir", "data/graph_store")
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # MultiDiGraph: multiple edges allowed between same (src,dst) pair
        # (different relation types). Edge key = relation type.
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()

        # alias → canonical node key, for fast fuzzy / alias resolution
        self._alias_index: Dict[str, str] = {}

        # community_id → GraphCommunity
        self.communities: Dict[str, GraphCommunity] = {}

        logger.info(
            "GraphStore v1.0 | dir=%s | leiden=%s | fuzzy=%s",
            self.persist_dir, _LEIDEN_AVAILABLE, _FUZZY_AVAILABLE,
        )

        # Charger automatiquement le snapshot persistant au démarrage.
        try:
            loaded = self.load()
            if not loaded:
                logger.info("GraphStore: fresh start (no persisted graph)")
        except Exception as exc:
            logger.warning("GraphStore auto-load failed: %s", exc)

    # ─────────────────────────────────────────────────────────────────
    # Insertion
    # ─────────────────────────────────────────────────────────────────
    def add_entities(self, entities: Sequence[GraphEntity]) -> int:
        """Add or merge entities. Returns number of *new* nodes created."""
        n_new = 0
        for ent in entities:
            if not ent.name:
                continue
            key = _canon_key(ent.name)
            if not key:
                continue

            if key in self.graph:
                self._merge_entity_into_node(key, ent)
            else:
                self._create_node(key, ent)
                n_new += 1

            # alias index
            self._alias_index[key] = key
            for alias in ent.aliases:
                self._alias_index[_canon_key(alias)] = key
        return n_new

    def _create_node(self, key: str, ent: GraphEntity) -> None:
        # NEW v1.0.1 : derive video_ids from BOTH ent.video_ids AND its mentions,
        # so callers don't have to set both sides explicitly.
        derived_videos = set(ent.video_ids) | {
            m.video_id for m in ent.mentions if m.video_id
        }
        self.graph.add_node(
            key,
            name=ent.name,
            type=normalize_entity_type(ent.type),
            description=ent.description or "",
            aliases=set(ent.aliases) | {ent.name},
            mentions=[m.as_dict() for m in ent.mentions],
            video_ids=derived_videos,
            n_mentions=len(ent.mentions),
        )

    def _merge_entity_into_node(self, key: str, ent: GraphEntity) -> None:
        node = self.graph.nodes[key]
        # type: keep most specific (non-OTHER wins)
        if node.get("type", "OTHER") == "OTHER" and ent.type != "OTHER":
            node["type"] = ent.type
        # description: longest wins, with concat if very different
        old_desc = node.get("description", "")
        if ent.description and ent.description not in old_desc:
            new_desc = (old_desc + "; " + ent.description).strip("; ")
            node["description"] = new_desc[:600]
        # aliases
        node["aliases"] = set(node.get("aliases", set())) | set(ent.aliases) | {ent.name}
        # mentions
        node["mentions"].extend(m.as_dict() for m in ent.mentions)
        # NEW v1.0.1 : also derive videos from mentions
        derived_videos = set(ent.video_ids) | {
            m.video_id for m in ent.mentions if m.video_id
        }
        node["video_ids"] = set(node.get("video_ids", set())) | derived_videos
        node["n_mentions"] = len(node["mentions"])

    def add_relations(self, relations: Sequence[GraphRelation]) -> int:
        """Add or merge relations. Returns number of *new* edges created."""
        n_new = 0
        for rel in relations:
            src_key = self._resolve_node(rel.src)
            dst_key = self._resolve_node(rel.dst)
            if not src_key or not dst_key or src_key == dst_key:
                continue
            rtype = normalize_relation_type(rel.type)

            # MultiDiGraph: edge key = relation type → max 1 edge per type per pair
            if self.graph.has_edge(src_key, dst_key, key=rtype):
                self._merge_relation_into_edge(src_key, dst_key, rtype, rel)
            else:
                self.graph.add_edge(
                    src_key, dst_key,
                    key=rtype,
                    type=rtype,
                    description=rel.description or "",
                    weight=float(rel.weight or 1.0),
                    evidence=[e.as_dict() for e in rel.evidence],
                    n_evidence=len(rel.evidence),
                )
                n_new += 1
        return n_new

    def _merge_relation_into_edge(
        self, src: str, dst: str, rtype: str, rel: GraphRelation
    ) -> None:
        edge = self.graph[src][dst][rtype]
        if rel.description and rel.description not in edge.get("description", ""):
            edge["description"] = (
                (edge.get("description", "") + " | " + rel.description).strip(" |")
            )[:600]
        edge["evidence"].extend(e.as_dict() for e in rel.evidence)
        edge["n_evidence"] = len(edge["evidence"])
        edge["weight"] = min(float(edge.get("weight", 1.0)) + 0.5, 5.0)

    # ─────────────────────────────────────────────────────────────────
    # Resolution (alias / fuzzy)
    # ─────────────────────────────────────────────────────────────────
    def _resolve_node(self, name: str, fuzzy: bool = True, threshold: int = 88) -> Optional[str]:
        """Return the canonical node key for a name, or None if not found."""
        if not name:
            return None
        key = _canon_key(name)
        if key in self.graph:
            return key
        if key in self._alias_index:
            return self._alias_index[key]
        if not fuzzy or not _FUZZY_AVAILABLE or self.graph.number_of_nodes() == 0:
            return None
        # fuzzy match against alias index keys
        match = _fuzzproc.extractOne(
            key, list(self._alias_index.keys()), scorer=fuzz.WRatio
        )
        if match and match[1] >= threshold:
            return self._alias_index[match[0]]
        return None

    # ─────────────────────────────────────────────────────────────────
    # Per-video views & deletion
    # ─────────────────────────────────────────────────────────────────
    def subgraph_for_video(self, video_id: str) -> nx.MultiDiGraph:
        nodes = [
            n for n, d in self.graph.nodes(data=True)
            if video_id in d.get("video_ids", set())
        ]
        return self.graph.subgraph(nodes).copy()

    # ─────────────────────────────────────────────────────────────────
    # ENRICHMENT — inject speaker & scene data directly from VideoStore
    # ─────────────────────────────────────────────────────────────────

    def enrich_from_video_store(
        self,
        video_store: Any,
        video_id: str,
    ) -> Dict[str, int]:
        """
        Scan VideoStore._meta for the given video and inject:
          • PERSON nodes for every distinct speaker label (with timestamps)
          • VIDEO node representing the video itself
          • speaks_in edges: PERSON → VIDEO
          • scene/visual nodes from visual chunk captions (CONCEPT type)

        This runs in O(chunks) with NO LLM calls, guaranteeing that speaker
        data is always in the graph regardless of entity-extraction quality.

        Returns dict with counts: {"speakers": N, "scenes": N, "new_nodes": N}
        """
        # FIX-1: normalise video_id — strip whitespace that can arrive from
        # URL-encoded paths (e.g. %20 prefix) or caller mistakes.
        video_id = (video_id or "").strip()
        chunk_ids = getattr(video_store, "_video_chunks", {}).get(video_id, set())
        meta_store = getattr(video_store, "_meta", {})

        speaker_chunks: Dict[str, List[Dict[str, Any]]] = {}  # speaker → list of meta
        scene_captions: List[Dict[str, Any]] = []

        for cid in chunk_ids:
            m = meta_store.get(cid, {})
            spk = str(m.get("speaker") or "").strip()
            chunk_type = m.get("chunk_type", "speech")

            if chunk_type == "speech" and spk and spk.lower() not in (
                "", "none", "unknown",
            ):
                speaker_chunks.setdefault(spk, []).append(m)

            elif chunk_type == "visual":
                text = (m.get("text") or "").strip()
                if text and len(text) > 10:
                    scene_captions.append({
                        "text": text[:200],
                        "start_time": m.get("start_time"),
                        "chunk_id": cid,
                    })

        new_nodes = 0

        # ── VIDEO node ────────────────────────────────────────────────
        vid_key = _canon_key(video_id)
        if vid_key not in self.graph:
            self.graph.add_node(
                vid_key,
                name=video_id,
                type="VIDEO",
                description=f"Video {video_id}",
                aliases={video_id},
                mentions=[],
                video_ids={video_id},
                n_mentions=1,
            )
            self._alias_index[vid_key] = vid_key
            new_nodes += 1
        else:
            node = self.graph.nodes[vid_key]
            node.setdefault("video_ids", set()).add(video_id)

        # ── SPEAKER nodes ─────────────────────────────────────────────
        speakers_added = 0
        for spk_label, chunks_list in speaker_chunks.items():
            key = _canon_key(spk_label)
            if not key:
                continue

            timestamps = sorted(
                float(m.get("start_time") or 0) for m in chunks_list
            )
            first_ts = timestamps[0] if timestamps else 0.0
            chunk_id_0 = chunks_list[0].get("chunk_id", "") if chunks_list else ""

            mention_dict = {
                "video_id": video_id,
                "chunk_id": chunk_id_0,
                "timestamp": first_ts,
                "speaker": spk_label,
                "confidence": 1.0,  # direct metadata — no LLM
            }

            if key not in self.graph:
                # Try to infer a human-readable name from the label
                display_name = spk_label
                if re.match(r"^speaker_(\d+)$", spk_label, re.IGNORECASE):
                    num = int(re.match(r"^speaker_(\d+)$", spk_label, re.IGNORECASE).group(1))
                    display_name = f"Speaker {num + 1}"  # 0-indexed → 1-indexed

                self.graph.add_node(
                    key,
                    name=display_name,
                    type="PERSON",
                    description=(
                        f"Speaker in video {video_id}. "
                        f"Speaks at {len(chunks_list)} segments "
                        f"(first at {int(first_ts)}s)."
                    ),
                    aliases={spk_label, display_name},
                    mentions=[mention_dict],
                    video_ids={video_id},
                    n_mentions=len(chunks_list),
                    # store all timestamps for timeline queries
                    speaker_timestamps=timestamps,
                )
                self._alias_index[key] = key
                self._alias_index[_canon_key(spk_label)] = key
                new_nodes += 1
                speakers_added += 1
            else:
                node = self.graph.nodes[key]
                node["n_mentions"] = node.get("n_mentions", 0) + len(chunks_list)
                node.setdefault("video_ids", set()).add(video_id)
                node.setdefault("mentions", []).append(mention_dict)
                node.setdefault("speaker_timestamps", []).extend(timestamps)

            # speaks_in edge: SPEAKER → VIDEO
            if not self.graph.has_edge(key, vid_key, "speaks_in"):
                self.graph.add_edge(
                    key, vid_key,
                    key="speaks_in",
                    type="speaks_in",
                    description=f"Speaks in {video_id}",
                    weight=float(len(chunks_list)),
                    evidence=[mention_dict],
                )

        # ── SCENE / VISUAL nodes (top 5 most informative) ────────────
        scenes_added = 0
        for sc in scene_captions[:5]:
            # Use first 40 chars of caption as node key
            short_cap = sc["text"][:40].strip()
            key = _canon_key(f"scene_{short_cap}")
            if key and key not in self.graph:
                self.graph.add_node(
                    key,
                    name=short_cap,
                    type="CONCEPT",
                    description=sc["text"],
                    aliases={short_cap},
                    mentions=[{
                        "video_id": video_id,
                        "chunk_id": sc["chunk_id"],
                        "timestamp": sc.get("start_time") or 0,
                    }],
                    video_ids={video_id},
                    n_mentions=1,
                )
                self._alias_index[key] = key
                new_nodes += 1
                scenes_added += 1
                # scene appears_in VIDEO
                if not self.graph.has_edge(key, vid_key, "appears_in"):
                    self.graph.add_edge(
                        key, vid_key,
                        key="appears_in",
                        type="appears_in",
                        description="Visual scene in video",
                        weight=1.0,
                        evidence=[],
                    )

        logger.info(
            "enrich_from_video_store | video=%s | speakers=%d | scenes=%d | new_nodes=%d",
            video_id, speakers_added, scenes_added, new_nodes,
        )
        return {"speakers": speakers_added, "scenes": scenes_added, "new_nodes": new_nodes}

    def get_speakers_for_video(
        self, video_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Return all PERSON nodes that have video_id in their video_ids set.
        Fast metadata lookup — used to answer 'how many speakers' questions.
        """
        speakers = []
        for n, d in self.graph.nodes(data=True):
            if (
                d.get("type") == "PERSON"
                and video_id in d.get("video_ids", set())
            ):
                speakers.append({
                    "key": n,
                    "name": d.get("name", n),
                    "n_mentions": d.get("n_mentions", 0),
                    "timestamps": sorted(d.get("speaker_timestamps", [])),
                })
        speakers.sort(key=lambda x: -x["n_mentions"])
        return speakers

    def delete_video(self, video_id: str) -> Dict[str, int]:
        """Remove all mentions/edges that belong to this video. Returns counts."""
        n_nodes_removed = 0
        n_edges_removed = 0

        # edges first
        edges_to_remove = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            new_ev = [e for e in d.get("evidence", []) if e.get("video_id") != video_id]
            if not new_ev:
                edges_to_remove.append((u, v, k))
            elif len(new_ev) != len(d.get("evidence", [])):
                d["evidence"] = new_ev
                d["n_evidence"] = len(new_ev)
        for u, v, k in edges_to_remove:
            self.graph.remove_edge(u, v, key=k)
            n_edges_removed += 1

        # nodes
        nodes_to_remove = []
        for n, d in self.graph.nodes(data=True):
            new_mentions = [m for m in d.get("mentions", []) if m.get("video_id") != video_id]
            new_videos = set(d.get("video_ids", set())) - {video_id}
            if not new_mentions and not new_videos:
                nodes_to_remove.append(n)
            else:
                d["mentions"] = new_mentions
                d["video_ids"] = new_videos
                d["n_mentions"] = len(new_mentions)
        for n in nodes_to_remove:
            # cleanup alias index
            for alias_key, target in list(self._alias_index.items()):
                if target == n:
                    self._alias_index.pop(alias_key, None)
            self.graph.remove_node(n)
            n_nodes_removed += 1

        # invalidate communities (they may reference removed nodes)
        if n_nodes_removed or n_edges_removed:
            self.communities = {}

        logger.info(
            "GraphStore.delete_video(%s) → removed %d nodes, %d edges",
            video_id, n_nodes_removed, n_edges_removed,
        )
        return {"nodes_removed": n_nodes_removed, "edges_removed": n_edges_removed}

    # ─────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────
    def save(self) -> None:
        t0 = time.time()
        # pickle = full fidelity
        with open(self.persist_dir / self.PICKLE_NAME, "wb") as f:
            pickle.dump(
                {
                    "graph": self.graph,
                    "alias_index": self._alias_index,
                    "communities": {cid: asdict(c) for cid, c in self.communities.items()},
                },
                f,
            )
        # json = portable view
        with open(self.persist_dir / self.JSON_NAME, "w", encoding="utf-8") as f:
            json.dump(self.to_json_dict(), f, ensure_ascii=False, indent=2, default=str)
        with open(self.persist_dir / self.COMM_NAME, "w", encoding="utf-8") as f:
            json.dump(
                {cid: c.as_dict() for cid, c in self.communities.items()},
                f, ensure_ascii=False, indent=2,
            )
        logger.info(
            "GraphStore.save | nodes=%d edges=%d communities=%d | %.2fs",
            self.graph.number_of_nodes(), self.graph.number_of_edges(),
            len(self.communities), time.time() - t0,
        )

    def load(self) -> bool:
        path = self.persist_dir / self.PICKLE_NAME
        if not path.exists():
            logger.info("GraphStore.load | no snapshot at %s", path)
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.graph = data.get("graph", nx.MultiDiGraph())
        self._alias_index = data.get("alias_index", {}) or {}
        if not self._alias_index:
            self._rebuild_alias_index()
        comm_dicts = data.get("communities", {})
        self.communities = {}
        for cid, cd in comm_dicts.items():
            cd.pop("size", None)  # recomputed
            self.communities[cid] = GraphCommunity(**cd)
            self.communities[cid].size = len(self.communities[cid].members)
        logger.info(
            "GraphStore.load | nodes=%d edges=%d communities=%d",
            self.graph.number_of_nodes(), self.graph.number_of_edges(),
            len(self.communities),
        )
        return True

    def _rebuild_alias_index(self) -> None:
        rebuilt: Dict[str, str] = {}
        for key, data in self.graph.nodes(data=True):
            rebuilt[_canon_key(data.get("name", key))] = key
            for alias in data.get("aliases", set()) or set():
                alias_key = _canon_key(alias)
                if alias_key:
                    rebuilt[alias_key] = key
        self._alias_index = rebuilt

    def to_json_dict(self) -> Dict[str, Any]:
        nodes = []
        for n, d in self.graph.nodes(data=True):
            nodes.append({
                "key": n,
                "name": d.get("name", n),
                "type": d.get("type", "OTHER"),
                "description": d.get("description", ""),
                "aliases": sorted(d.get("aliases", set())),
                "video_ids": sorted(d.get("video_ids", set())),
                "n_mentions": d.get("n_mentions", 0),
            })
        edges = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            edges.append({
                "src": u,
                "dst": v,
                "type": k,
                "description": d.get("description", ""),
                "weight": d.get("weight", 1.0),
                "n_evidence": d.get("n_evidence", 0),
            })
        return {"nodes": nodes, "edges": edges}

    # ─────────────────────────────────────────────────────────────────
    # Community detection (Leiden / fallback)
    # ─────────────────────────────────────────────────────────────────
    def detect_communities(
        self,
        llm_client: Any = None,
        max_communities: int = 50,
        min_size: int = 2,
        summarize: bool = True,
    ) -> List[GraphCommunity]:
        """Run community detection then (optionally) LLM-summarize each community."""
        t0 = time.time()
        if self.graph.number_of_nodes() == 0:
            self.communities = {}
            return []

        partition = self._run_partition()
        # group nodes
        groups: Dict[int, List[str]] = defaultdict(list)
        for node, cid in partition.items():
            groups[cid].append(node)

        # filter + sort by size desc
        valid = [
            (cid, members) for cid, members in groups.items()
            if len(members) >= min_size
        ]
        valid.sort(key=lambda kv: -len(kv[1]))
        valid = valid[:max_communities]

        new_communities: Dict[str, GraphCommunity] = {}
        for i, (cid, members) in enumerate(valid):
            cid_str = f"community_{i:04d}"
            keywords = self._community_keywords(members)
            comm = GraphCommunity(
                community_id=cid_str,
                members=members,
                title=", ".join(keywords[:3]) or f"Community {i}",
                summary="",
                keywords=keywords,
                size=len(members),
            )
            new_communities[cid_str] = comm

        self.communities = new_communities
        logger.info(
            "GraphStore.detect_communities | %d partitions → %d kept | %.2fs",
            len(groups), len(self.communities), time.time() - t0,
        )

        if summarize and llm_client is not None and self.communities:
            self._summarize_communities(llm_client)

        return list(self.communities.values())

    def _run_partition(self) -> Dict[str, int]:
        """Returns {node_key: community_id_int}."""
        # Convert to undirected for community detection
        und = nx.Graph()
        for u, v, d in self.graph.edges(data=True):
            w = float(d.get("weight", 1.0))
            if und.has_edge(u, v):
                und[u][v]["weight"] += w
            else:
                und.add_edge(u, v, weight=w)
        for n in self.graph.nodes():
            if n not in und:
                und.add_node(n)

        if _LEIDEN_AVAILABLE and und.number_of_edges() > 0:
            try:
                # Build igraph in the same node order as 'und'
                node_list = list(und.nodes())
                idx = {n: i for i, n in enumerate(node_list)}
                edges_ig = [(idx[u], idx[v]) for u, v in und.edges()]
                weights = [und[u][v]["weight"] for u, v in und.edges()]
                g = ig.Graph(n=len(node_list), edges=edges_ig, directed=False)
                g.es["weight"] = weights
                part = la.find_partition(
                    g, la.RBConfigurationVertexPartition,
                    weights="weight", resolution_parameter=1.0,
                    seed=42,
                )
                membership = part.membership
                return {node_list[i]: cid for i, cid in enumerate(membership)}
            except Exception as exc:
                logger.warning("Leiden failed (%s) → greedy fallback", exc)

        # Fallback: greedy modularity
        try:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = list(greedy_modularity_communities(und, weight="weight"))
            return {n: i for i, c in enumerate(comms) for n in c}
        except Exception:
            # ultimate fallback: connected components
            return {n: i for i, comp in enumerate(nx.connected_components(und)) for n in comp}

    def _community_keywords(self, members: List[str]) -> List[str]:
        """Extract the top keywords for a community: most-mentioned named entities."""
        scored: List[Tuple[str, int]] = []
        for n in members:
            d = self.graph.nodes[n]
            scored.append((d.get("name", n), int(d.get("n_mentions", 1))))
        scored.sort(key=lambda kv: -kv[1])
        return [name for name, _ in scored[:8]]

    def _summarize_communities(self, llm_client: Any) -> None:
        """Use the LLM (chat_json or answer_question) to write a 2-4 sentence report."""
        for cid, comm in self.communities.items():
            try:
                ctx = self._community_context(comm, max_chars=1800)
                prompt = (
                    "You are a research analyst. Write a concise 2-4 sentence "
                    "summary describing the COMMON THEME shared by the entities "
                    "below. Be specific. Mention the most important entities. "
                    "Output plain text only, no headings, no bullet points.\n\n"
                    f"ENTITIES & RELATIONS:\n{ctx}\n\nSUMMARY:"
                )
                txt = self._llm_plain(llm_client, prompt, max_tokens=180)
                comm.summary = (txt or "").strip()[:800]
            except Exception as exc:
                logger.warning("Summarize community %s failed: %s", cid, exc)

    def _community_context(self, comm: GraphCommunity, max_chars: int = 1800) -> str:
        lines = []
        for m in comm.members[:25]:
            d = self.graph.nodes[m]
            lines.append(
                f"- [{d.get('type','OTHER')}] {d.get('name', m)}"
                + (f" — {_short(d.get('description',''), 120)}" if d.get("description") else "")
            )
        # a few edges
        rel_lines = []
        sub = self.graph.subgraph(comm.members)
        for u, v, k, d in list(sub.edges(keys=True, data=True))[:25]:
            rel_lines.append(
                f"- {self.graph.nodes[u]['name']} --{k}--> {self.graph.nodes[v]['name']}"
            )
        out = "Entities:\n" + "\n".join(lines)
        if rel_lines:
            out += "\n\nRelations:\n" + "\n".join(rel_lines)
        return out[:max_chars]

    @staticmethod
    def _llm_plain(llm_client: Any, prompt: str, max_tokens: int = 200) -> str:
        """Call the LLM via the most flexible available interface."""
        if llm_client is None:
            return ""
        # 1) chat_json helper (added in patched llm_client)
        for fn_name in ("chat_plain", "complete", "chat"):
            fn = getattr(llm_client, fn_name, None)
            if callable(fn):
                try:
                    out = fn(prompt=prompt, num_predict=max_tokens, temperature=0.2)
                    if isinstance(out, str):
                        return out
                    if isinstance(out, dict):
                        return out.get("text") or out.get("response") or ""
                except Exception:
                    pass
        # 2) module-level synthesize_answer fallback
        try:
            from core import llm_client as lc
            res = lc.synthesize_answer(
                question=prompt, results=[], language="en",
                max_items=0, num_predict=max_tokens,
            )
            if hasattr(res, "answer"):
                return res.answer
            if isinstance(res, dict):
                return res.get("answer", "")
        except Exception:
            pass
        return ""

    # ─────────────────────────────────────────────────────────────────
    # LOCAL SEARCH — entity-centric neighborhood
    # ─────────────────────────────────────────────────────────────────
    def find_seed_entities(
        self, query: str, top_k: int = 6, threshold: int = 70,
    ) -> List[str]:
        """Return canonical node keys most similar to `query`."""
        if not query or self.graph.number_of_nodes() == 0:
            return []
        q = _canon_key(query)
        seeds: List[Tuple[str, int]] = []

        # Detect speaker-related queries to boost PERSON nodes
        _speaker_query = bool(re.search(
            r"\b(speaker|speakers|person|people|who|voice|voices|"
            r"interviewee|presenter|narrator|host|guest|talk|said|spoke)\b",
            query, re.IGNORECASE,
        ))

        # 1) exact-token + type-boost match
        q_tokens = set(re.findall(r"[\w\u0600-\u06FF]+", q))
        for n, d in self.graph.nodes(data=True):
            name_low = _canon_key(d.get("name", n))
            aliases_low = {_canon_key(a) for a in d.get("aliases", set())}
            name_tokens = set(re.findall(r"[\w\u0600-\u06FF]+", name_low))
            score = 0
            if name_low and (name_low in q or q in name_low):
                score = max(score, 95)
            # token-level overlap (catches 'BB84' inside 'Tell me about BB84')
            shared = q_tokens & name_tokens
            if shared:
                score = max(score, 75 + min(15, 5 * len(shared)))
            for tok in q_tokens:
                if tok and len(tok) >= 3 and (
                    tok in name_low or any(tok in a for a in aliases_low)
                ):
                    score = max(score, 70)
            # BOOST: speaker queries always pull in all PERSON nodes
            if _speaker_query and d.get("type") == "PERSON":
                score = max(score, 80)
            if score:
                seeds.append((n, score))

        # 2) fuzzy match
        if _FUZZY_AVAILABLE and len(seeds) < top_k:
            choices = {n: _canon_key(d.get("name", n)) for n, d in self.graph.nodes(data=True)}
            res = _fuzzproc.extract(
                q, choices, scorer=fuzz.WRatio, limit=top_k * 2,
            )
            for _matched_name, score, key in res:
                if score >= threshold:
                    seeds.append((key, int(score)))

        # dedup keep highest score
        best: Dict[str, int] = {}
        for k, s in seeds:
            best[k] = max(best.get(k, 0), s)
        ranked = sorted(best.items(), key=lambda kv: -kv[1])[:top_k]
        return [k for k, _ in ranked]

    def local_search(
        self,
        query: Optional[str] = None,
        seed_entities: Optional[List[str]] = None,
        hops: int = 1,
        max_nodes: int = 30,
        video_id: Optional[str] = None,
    ) -> GraphSearchResult:
        """Entity-centric search: seeds → neighborhood → context."""
        seeds: List[str] = []
        if seed_entities:
            for s in seed_entities:
                k = self._resolve_node(s)
                if k:
                    seeds.append(k)
        if not seeds and query:
            seeds = self.find_seed_entities(query, top_k=6)

        if not seeds:
            return GraphSearchResult(mode="local", seed_entities=[], context_text="")

        # k-hop neighborhood
        visited: Set[str] = set(seeds)
        frontier = set(seeds)
        for _ in range(max(hops, 1)):
            new_frontier: Set[str] = set()
            for n in frontier:
                if n not in self.graph:
                    continue
                new_frontier |= set(self.graph.successors(n))
                new_frontier |= set(self.graph.predecessors(n))
            new_frontier -= visited
            visited |= new_frontier
            frontier = new_frontier
            if len(visited) >= max_nodes:
                break

        # filter by video_id if requested
        if video_id:
            visited = {
                n for n in visited
                if video_id in self.graph.nodes[n].get("video_ids", set())
            }

        # rank: seeds first, then by mention count
        def _rank(n: str) -> tuple:
            return (0 if n in seeds else 1, -int(self.graph.nodes[n].get("n_mentions", 0)))

        ordered = sorted(visited, key=_rank)[:max_nodes]
        node_set = set(ordered)

        # collect edges within the neighborhood
        nodes_out: List[Dict[str, Any]] = []
        chunk_ids: Set[str] = set()
        for n in ordered:
            d = self.graph.nodes[n]
            nodes_out.append({
                "key": n,
                "name": d.get("name", n),
                "type": d.get("type", "OTHER"),
                "description": d.get("description", ""),
                "n_mentions": d.get("n_mentions", 0),
                "is_seed": n in seeds,
            })
            for m in d.get("mentions", []):
                cid = m.get("chunk_id")
                if cid and (not video_id or m.get("video_id") == video_id):
                    chunk_ids.add(cid)

        edges_out: List[Dict[str, Any]] = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            if u in node_set and v in node_set:
                edges_out.append({
                    "src": self.graph.nodes[u].get("name", u),
                    "dst": self.graph.nodes[v].get("name", v),
                    "src_key": u, "dst_key": v,
                    "type": k,
                    "description": d.get("description", ""),
                    "weight": d.get("weight", 1.0),
                })
                for ev in d.get("evidence", []):
                    cid = ev.get("chunk_id")
                    if cid and (not video_id or ev.get("video_id") == video_id):
                        chunk_ids.add(cid)

        ctx_text = self._format_local_context(nodes_out, edges_out)

        return GraphSearchResult(
            mode="local",
            seed_entities=[self.graph.nodes[s].get("name", s) for s in seeds],
            nodes=nodes_out,
            edges=edges_out,
            chunk_ids=sorted(chunk_ids),
            context_text=ctx_text,
        )

    @staticmethod
    def _format_local_context(
        nodes: List[Dict[str, Any]], edges: List[Dict[str, Any]],
    ) -> str:
        if not nodes:
            return ""
        head = "## ENTITIES (knowledge graph neighborhood)\n"
        ent_lines = []
        for n in nodes[:25]:
            tag = "★" if n.get("is_seed") else "•"
            desc = _short(n.get("description", ""), 140)
            ent_lines.append(
                f"{tag} [{n['type']}] **{n['name']}** "
                + (f"— {desc}" if desc else "")
            )
        out = head + "\n".join(ent_lines)
        if edges:
            out += "\n\n## RELATIONS\n"
            out += "\n".join(
                f"• {e['src']} —[{e['type']}]→ {e['dst']}"
                + (f" — {_short(e['description'], 100)}" if e.get("description") else "")
                for e in edges[:25]
            )
        return out

    # ─────────────────────────────────────────────────────────────────
    # ─────────────────────────────────────────────────────────────────
    # MULTI-HOP GRAPH SEARCH
    # Follows directed edges across the graph for multi-step reasoning:
    #   "What did the speaker say AFTER discussing X?"
    #   "What caused Y and what were the consequences?"
    # ─────────────────────────────────────────────────────────────────
    def multi_hop_graph_search(
        self,
        query: str,
        seed_entities: Optional[List[str]] = None,
        relation_filter: Optional[List[str]] = None,
        hops: int = 3,
        max_nodes_per_hop: int = 8,
        video_id: Optional[str] = None,
        direction: str = "both",          # "forward" | "backward" | "both"
        temporal_order: bool = True,      # sort collected chunks by start_time
    ) -> "GraphSearchResult":
        """
        Multi-hop graph traversal for complex reasoning questions.

        Unlike local_search (which expands uniformly in all directions),
        this method follows relation-typed edges hop by hop, building a
        chain of evidence. Each hop collects the chunks linked to nodes
        at that depth, giving the LLM a temporally-ordered evidence trail.

        Args:
            query:           The original question (used to find seeds if
                             seed_entities is None).
            seed_entities:   Starting entity names.  Auto-resolved from
                             the query if not provided.
            relation_filter: Only follow edges of these types, e.g.
                             ["precedes", "causes", "defined_as"].
                             None → follow all relation types.
            hops:            Maximum depth to traverse.
            max_nodes_per_hop: Cap on new nodes explored per hop.
            video_id:        Restrict to chunks from this video.
            direction:       "forward"  → follow successors only (A→B→C)
                             "backward" → follow predecessors only (C←B←A)
                             "both"     → expand in both directions
            temporal_order:  Sort final chunk_ids by start_time ascending.

        Returns:
            GraphSearchResult with:
                - nodes: all visited nodes (with hop_depth label)
                - edges: all traversed edges
                - chunk_ids: evidence chunks in temporal order
                - context_text: formatted multi-hop chain for LLM
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            return GraphSearchResult(mode="multi_hop", context_text="")

        # 1. Resolve seed nodes
        seeds: List[str] = []
        if seed_entities:
            for s in seed_entities:
                k = self._resolve_node(s)
                if k:
                    seeds.append(k)
        if not seeds and query:
            seeds = self.find_seed_entities(query, top_k=4)
        if not seeds:
            logger.info("multi_hop_graph_search: no seed entities found for %r", query[:60])
            return GraphSearchResult(mode="multi_hop", context_text="")

        logger.info(
            "multi_hop_graph_search: seeds=%s hops=%d rel_filter=%s dir=%s",
            [self.graph.nodes[s].get("name", s) for s in seeds],
            hops, relation_filter, direction,
        )

        # 2. BFS hop-by-hop, collecting nodes per hop level
        hop_levels: List[Dict[str, int]] = []  # [{node_key: hop_depth}, ...]
        visited: Set[str] = set(seeds)
        frontier: Set[str] = set(seeds)
        all_nodes: Dict[str, int] = {s: 0 for s in seeds}  # node_key → hop_depth

        for hop in range(hops):
            new_frontier: Set[str] = set()
            for n in frontier:
                if n not in self.graph:
                    continue
                # Forward edges (n → successor)
                if direction in ("forward", "both"):
                    for _, v, rel_type, _ in self.graph.out_edges(n, keys=True, data=True):
                        if relation_filter and rel_type not in relation_filter:
                            continue
                        if v not in visited:
                            new_frontier.add(v)
                # Backward edges (predecessor → n)
                if direction in ("backward", "both"):
                    for u, _, rel_type, _ in self.graph.in_edges(n, keys=True, data=True):
                        if relation_filter and rel_type not in relation_filter:
                            continue
                        if u not in visited:
                            new_frontier.add(u)

            # Apply video_id filter and cap per hop
            if video_id:
                new_frontier = {
                    n for n in new_frontier
                    if video_id in self.graph.nodes[n].get("video_ids", set())
                }
            # Cap by mention count (most-mentioned = most important)
            if len(new_frontier) > max_nodes_per_hop:
                new_frontier = set(sorted(
                    new_frontier,
                    key=lambda n: -int(self.graph.nodes[n].get("n_mentions", 0)),
                )[:max_nodes_per_hop])

            if not new_frontier:
                logger.debug("multi_hop_graph_search: frontier empty at hop %d — stopping", hop + 1)
                break

            for n in new_frontier:
                all_nodes[n] = hop + 1
            visited |= new_frontier
            frontier = new_frontier
            hop_levels.append({n: hop + 1 for n in new_frontier})

        # 3. Collect evidence chunks from all visited nodes, with timestamps
        chunk_ids_with_ts: List[Tuple[float, str]] = []   # (start_time, chunk_id)
        nodes_out: List[Dict[str, Any]] = []
        chunk_ids_seen: Set[str] = set()

        for n, depth in sorted(all_nodes.items(), key=lambda kv: kv[1]):
            d = self.graph.nodes[n]
            nodes_out.append({
                "key": n,
                "name": d.get("name", n),
                "type": d.get("type", "OTHER"),
                "description": d.get("description", ""),
                "n_mentions": d.get("n_mentions", 0),
                "hop_depth": depth,
                "is_seed": depth == 0,
            })
            for mention in d.get("mentions", []):
                cid = mention.get("chunk_id")
                ts  = float(mention.get("timestamp", 0.0))
                if cid and cid not in chunk_ids_seen:
                    if not video_id or mention.get("video_id") == video_id:
                        chunk_ids_with_ts.append((ts, cid))
                        chunk_ids_seen.add(cid)

        # 4. Collect edges within visited subgraph
        node_set = set(all_nodes.keys())
        edges_out: List[Dict[str, Any]] = []
        for u, v, rel_type, edata in self.graph.edges(keys=True, data=True):
            if u in node_set and v in node_set:
                if relation_filter and rel_type not in relation_filter:
                    continue
                edge_entry: Dict[str, Any] = {
                    "src": self.graph.nodes[u].get("name", u),
                    "dst": self.graph.nodes[v].get("name", v),
                    "src_key": u, "dst_key": v,
                    "type": rel_type,
                    "weight": edata.get("weight", 1.0),
                    "src_depth": all_nodes.get(u, 0),
                    "dst_depth": all_nodes.get(v, 0),
                }
                edges_out.append(edge_entry)
                # Also collect chunk_ids from edge evidence
                for ev in edata.get("evidence", []):
                    cid = ev.get("chunk_id")
                    ts  = float(ev.get("timestamp", 0.0))
                    if cid and cid not in chunk_ids_seen:
                        if not video_id or ev.get("video_id") == video_id:
                            chunk_ids_with_ts.append((ts, cid))
                            chunk_ids_seen.add(cid)

        # 5. Sort chunks temporally
        if temporal_order:
            chunk_ids_with_ts.sort(key=lambda x: x[0])
        final_chunk_ids = [cid for _, cid in chunk_ids_with_ts]

        # 6. Build context string for LLM — shows the reasoning chain
        ctx = self._format_multi_hop_context(query, nodes_out, edges_out, hops)

        logger.info(
            "multi_hop_graph_search done: %d nodes, %d edges, %d chunks across %d hops",
            len(nodes_out), len(edges_out), len(final_chunk_ids), hops,
        )
        return GraphSearchResult(
            mode="multi_hop",
            seed_entities=[self.graph.nodes[s].get("name", s) for s in seeds],
            nodes=nodes_out,
            edges=edges_out,
            chunk_ids=final_chunk_ids,
            context_text=ctx,
        )

    def _format_multi_hop_context(
        self,
        query: str,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        max_hops: int,
    ) -> str:
        """Format multi-hop traversal as a reasoning chain for the LLM."""
        if not nodes:
            return ""
        lines = ["## MULTI-HOP KNOWLEDGE GRAPH CHAIN"]
        lines.append(f"Query: {query[:120]}")
        lines.append("")

        # Group nodes by hop depth
        max_depth = max((n.get("hop_depth", 0) for n in nodes), default=0)
        for depth in range(max_depth + 1):
            depth_nodes = [n for n in nodes if n.get("hop_depth", 0) == depth]
            if not depth_nodes:
                continue
            label = "SEEDS (starting entities)" if depth == 0 else f"HOP {depth} (connected entities)"
            lines.append(f"### {label}")
            for n in depth_nodes[:10]:
                tag = "★" if n.get("is_seed") else "•"
                desc = (n.get("description") or "")[:120]
                lines.append(
                    f"  {tag} [{n['type']}] **{n['name']}**"
                    + (f" — {desc}" if desc else "")
                )

        if edges:
            lines.append("")
            lines.append("### REASONING CHAIN (relations followed)")
            for e in edges[:20]:
                lines.append(
                    f"  {e['src']} —[{e['type']}]→ {e['dst']}"
                )

        return "\n".join(lines)

    # GLOBAL SEARCH — community-summary-centric
    # ─────────────────────────────────────────────────────────────────
    def global_search(
        self, query: str, top_k: int = 3, video_id: Optional[str] = None,
    ) -> GraphSearchResult:
        """Rank communities by relevance to query, return their summaries."""
        if not self.communities:
            return GraphSearchResult(mode="global", context_text="")

        q_tokens = set(re.findall(r"[\w\u0600-\u06FF]+", _canon_key(query)))
        ranked: List[Tuple[float, GraphCommunity]] = []
        for comm in self.communities.values():
            if video_id:
                # skip communities with no node from this video
                in_video = any(
                    video_id in self.graph.nodes[m].get("video_ids", set())
                    for m in comm.members if m in self.graph
                )
                if not in_video:
                    continue
            score = self._score_community(comm, q_tokens)
            comm.score = score
            ranked.append((score, comm))

        ranked.sort(key=lambda kv: -kv[0])
        top = [c for s, c in ranked[:top_k] if s > 0] or [c for _, c in ranked[:top_k]]

        ctx = self._format_global_context(query, top)
        chunk_ids: Set[str] = set()
        for comm in top:
            for m in comm.members:
                if m not in self.graph:
                    continue
                for mention in self.graph.nodes[m].get("mentions", []):
                    if not video_id or mention.get("video_id") == video_id:
                        if mention.get("chunk_id"):
                            chunk_ids.add(mention["chunk_id"])

        return GraphSearchResult(
            mode="global",
            seed_entities=[],
            nodes=self._nodes_from_communities(top, video_id),
            edges=self._edges_from_communities(top, video_id),
            chunk_ids=sorted(chunk_ids),
            communities=[c.as_dict() for c in top],
            context_text=ctx,
        )

    def _nodes_from_communities(
        self,
        communities: list,
        video_id: Optional[str] = None,
        max_nodes: int = 30,
    ) -> list:
        """Build a meaningful node list from community members for global search."""
        seen: set = set()
        nodes_out: list = []
        for comm in communities:
            for m in comm.members:
                if m in seen or m not in self.graph:
                    continue
                seen.add(m)
                d = self.graph.nodes[m]
                if video_id:
                    if video_id not in d.get("video_ids", set()):
                        continue
                nodes_out.append({
                    "key": m,
                    "name": d.get("name", m),
                    "type": d.get("type", "OTHER"),
                    "description": d.get("description", ""),
                    "n_mentions": d.get("n_mentions", 0),
                    "is_seed": False,
                })
                if len(nodes_out) >= max_nodes:
                    return nodes_out
        return nodes_out

    def _edges_from_communities(
        self,
        communities: list,
        video_id: Optional[str] = None,
        max_edges: int = 20,
    ) -> list:
        """Build an edge list from relations inside the community member set."""
        member_set: set = set()
        for comm in communities:
            member_set.update(m for m in comm.members if m in self.graph)

        edges_out: list = []
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            if u not in member_set or v not in member_set:
                continue
            if video_id:
                evidence = d.get("evidence", [])
                if not any(ev.get("video_id") == video_id for ev in evidence):
                    continue
            edges_out.append({
                "src": self.graph.nodes[u].get("name", u),
                "dst": self.graph.nodes[v].get("name", v),
                "src_key": u,
                "dst_key": v,
                "type": k,
                "description": d.get("description", ""),
                "weight": d.get("weight", 1.0),
            })
            if len(edges_out) >= max_edges:
                break
        return edges_out

    def _score_community(self, comm: GraphCommunity, q_tokens: Set[str]) -> float:
        if not q_tokens:
            return float(comm.size)
        text_pool = " ".join([
            comm.title or "", comm.summary or "",
            " ".join(comm.keywords or []),
        ]).lower()
        text_pool += " " + " ".join(
            self.graph.nodes[m].get("name", "").lower()
            for m in comm.members if m in self.graph
        )
        toks_in_text = set(re.findall(r"[\w\u0600-\u06FF]+", text_pool))
        overlap = len(q_tokens & toks_in_text)
        return float(overlap) + 0.05 * float(comm.size)

    @staticmethod
    def _format_global_context(query: str, comms: List[GraphCommunity]) -> str:
        if not comms:
            return ""
        out = ["## THEMATIC COMMUNITIES (knowledge graph)"]
        for c in comms:
            out.append(
                f"\n### {c.title or c.community_id}  (size={c.size})"
                f"\n{c.summary or '(no summary)'}\n"
                f"_keywords: {', '.join(c.keywords[:8])}_"
            )
        return "\n".join(out)

    # ─────────────────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────────────────
    def stats(self) -> Dict[str, Any]:
        type_counts = Counter(d.get("type", "OTHER") for _, d in self.graph.nodes(data=True))
        rel_counts = Counter(k for _, _, k in self.graph.edges(keys=True))
        videos: Set[str] = set()
        for _, d in self.graph.nodes(data=True):
            videos |= set(d.get("video_ids", set()))
        return {
            "n_nodes": self.graph.number_of_nodes(),
            "n_edges": self.graph.number_of_edges(),
            "n_communities": len(self.communities),
            "n_videos": len(videos),
            "videos": sorted(videos),
            "entity_types": dict(type_counts),
            "relation_types": dict(rel_counts),
            "leiden_available": _LEIDEN_AVAILABLE,
            "fuzzy_available": _FUZZY_AVAILABLE,
        }