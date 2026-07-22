"""
core/graph_contracts.py — GraphRAG data contracts (v1.0)
=========================================================
Canonical dataclasses used across graph_extractor, graph_builder,
graph_store, graph_search.

Design principles
-----------------
- Immutable canonical names for entity types (so the LLM and the
  store never disagree).
- Mentions store ENOUGH context to (a) jump to the timestamp,
  (b) display the speaker, (c) cite back to the chunk.
- Relations are directional: src --type--> dst.
- Communities have a stable id (community_<int>) and a
  summary that the global search will use as context.

Author : Spiral father, 2026.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


# ──────────────────────────────────────────────────────────────────────
# Entity / Relation type vocabularies
# ──────────────────────────────────────────────────────────────────────

#: Canonical entity types accepted by the extractor.
#: Any other type returned by the LLM is normalized to "OTHER".
ENTITY_TYPES: tuple = (
    "PERSON",       # named human (Alice Chen, Sam Altman, ...)
    "ORG",          # organization, company, university, lab
    "PLACE",        # geographical location, country, city, building
    "EVENT",        # conference, summit, war, election, launch
    "PRODUCT",      # commercial product, software, model, dataset
    "TECH",         # technology / method / algorithm (RSA, BB84, RAG, FAISS)
    "CONCEPT",      # abstract idea, theory, principle, framework
    "WORK",         # paper, book, video, song, film, episode
    "DATE",         # explicit date / period (2024, "last summer")
    "OTHER",
)

#: Canonical relation types. The extractor is asked to use
#: free-form labels; we normalize to these via simple synonym mapping.
RELATION_TYPES: tuple = (
    "discusses",          # speaker discusses concept
    "mentions",           # weak mention
    "supports",           # explicit support / advocacy
    "opposes",            # explicit disagreement
    "compares_to",        # X is compared to Y
    "part_of",            # X is part of Y
    "located_in",         # X is located in Y
    "created_by",         # work created by entity
    "works_at",           # person works at org
    "uses",               # X uses Y (tool / technology)
    "causes",             # causal relation
    "precedes",           # temporal precedence
    "related_to",         # generic catch-all
)

#: Synonyms → canonical relation. Used to normalize raw LLM output.
RELATION_SYNONYMS: Dict[str, str] = {
    "talks_about": "discusses", "speaks_about": "discusses", "covers": "discusses",
    "endorses": "supports", "advocates": "supports", "agrees_with": "supports",
    "disagrees_with": "opposes", "criticizes": "opposes", "rejects": "opposes",
    "vs": "compares_to", "versus": "compares_to", "compared_with": "compares_to",
    "belongs_to": "part_of", "member_of": "part_of", "subset_of": "part_of",
    "in": "located_in", "based_in": "located_in",
    "authored_by": "created_by", "developed_by": "created_by", "made_by": "created_by",
    "employed_at": "works_at", "founder_of": "works_at",
    "leverages": "uses", "applies": "uses", "based_on": "uses",
    "leads_to": "causes", "results_in": "causes",
    "before": "precedes", "comes_before": "precedes",
}

ENTITY_TYPE_SYNONYMS: Dict[str, str] = {
    "PEOPLE": "PERSON", "HUMAN": "PERSON", "INDIVIDUAL": "PERSON",
    "ORGANIZATION": "ORG", "COMPANY": "ORG", "UNIVERSITY": "ORG", "LAB": "ORG",
    "LOCATION": "PLACE", "GPE": "PLACE", "CITY": "PLACE", "COUNTRY": "PLACE",
    "MISC": "OTHER", "UNKNOWN": "OTHER",
    "TECHNOLOGY": "TECH", "METHOD": "TECH", "ALGORITHM": "TECH", "FRAMEWORK": "TECH",
    "IDEA": "CONCEPT", "THEORY": "CONCEPT", "PRINCIPLE": "CONCEPT",
    "PAPER": "WORK", "BOOK": "WORK", "ARTICLE": "WORK",
    "TIME": "DATE", "PERIOD": "DATE", "YEAR": "DATE",
}


def normalize_entity_type(raw: str) -> str:
    """Normalize a raw LLM-returned entity type to the canonical vocabulary."""
    if not raw:
        return "OTHER"
    raw_up = str(raw).strip().upper()
    if raw_up in ENTITY_TYPES:
        return raw_up
    return ENTITY_TYPE_SYNONYMS.get(raw_up, "OTHER")


def normalize_relation_type(raw: str) -> str:
    """Normalize a raw LLM-returned relation label to the canonical vocabulary."""
    if not raw:
        return "related_to"
    raw_low = str(raw).strip().lower().replace(" ", "_").replace("-", "_")
    if raw_low in RELATION_TYPES:
        return raw_low
    return RELATION_SYNONYMS.get(raw_low, "related_to")


# ──────────────────────────────────────────────────────────────────────
# Mentions / Evidence
# ──────────────────────────────────────────────────────────────────────

@dataclass
class EntityMention:
    """
    One occurrence of an entity in the corpus.

    Stored on the node so we can jump to *every* timestamp where the
    entity was mentioned, and cite the source chunk.
    """
    video_id: str
    chunk_id: str
    timestamp: float          # start_time in seconds
    end_time: Optional[float] = None
    speaker: Optional[str] = None
    language: Optional[str] = None
    raw_surface: str = ""     # the exact string used by the speaker
    confidence: float = 1.0   # extractor confidence

    def as_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "chunk_id": self.chunk_id,
            "timestamp": self.timestamp,
            "end_time": self.end_time,
            "speaker": self.speaker,
            "language": self.language,
            "raw_surface": self.raw_surface,
            "confidence": self.confidence,
        }


@dataclass
class RelationEvidence:
    """One chunk providing evidence for a (src, type, dst) relation."""
    video_id: str
    chunk_id: str
    timestamp: float
    end_time: Optional[float] = None
    speaker: Optional[str] = None
    snippet: str = ""         # short text snippet supporting the relation
    confidence: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "chunk_id": self.chunk_id,
            "timestamp": self.timestamp,
            "end_time": self.end_time,
            "speaker": self.speaker,
            "snippet": self.snippet,
            "confidence": self.confidence,
        }


# ──────────────────────────────────────────────────────────────────────
# Entity & Relation
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GraphEntity:
    """A node in the knowledge graph."""
    name: str                                  # canonical name, e.g. "Alice Chen"
    type: str = "OTHER"                        # one of ENTITY_TYPES
    description: str = ""                      # short LLM-generated description
    aliases: Set[str] = field(default_factory=set)
    mentions: List[EntityMention] = field(default_factory=list)
    video_ids: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.type = normalize_entity_type(self.type)
        self.name = self.name.strip()

    def add_mention(self, mention: EntityMention) -> None:
        self.mentions.append(mention)
        self.video_ids.add(mention.video_id)

    def merge(self, other: "GraphEntity") -> None:
        """Merge another entity (same canonical name) into this one."""
        if not self.description and other.description:
            self.description = other.description
        elif other.description and other.description not in self.description:
            self.description = f"{self.description}; {other.description}"[:600]
        self.aliases |= other.aliases
        self.mentions.extend(other.mentions)
        self.video_ids |= other.video_ids

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "aliases": sorted(self.aliases),
            "mentions": [m.as_dict() for m in self.mentions],
            "video_ids": sorted(self.video_ids),
            "n_mentions": len(self.mentions),
        }


@dataclass
class GraphRelation:
    """A directed edge in the knowledge graph."""
    src: str
    dst: str
    type: str = "related_to"
    description: str = ""
    evidence: List[RelationEvidence] = field(default_factory=list)
    weight: float = 1.0

    def __post_init__(self) -> None:
        self.type = normalize_relation_type(self.type)
        self.src = self.src.strip()
        self.dst = self.dst.strip()

    def add_evidence(self, ev: RelationEvidence) -> None:
        self.evidence.append(ev)
        # weight grows with number of supporting evidence (capped)
        self.weight = min(1.0 + 0.5 * (len(self.evidence) - 1), 5.0)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "src": self.src,
            "dst": self.dst,
            "type": self.type,
            "description": self.description,
            "weight": self.weight,
            "evidence": [e.as_dict() for e in self.evidence],
            "n_evidence": len(self.evidence),
        }


# ──────────────────────────────────────────────────────────────────────
# Communities
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GraphCommunity:
    """
    A community of related entities, detected via Leiden algorithm.
    The LLM-generated `summary` is used by GLOBAL search.
    """
    community_id: str
    members: List[str] = field(default_factory=list)   # entity names
    title: str = ""                                    # short label
    summary: str = ""                                  # 2-4 sentence thematic summary
    keywords: List[str] = field(default_factory=list)
    size: int = 0
    score: float = 0.0       # used at query time to rank communities

    def as_dict(self) -> Dict[str, Any]:
        return {
            "community_id": self.community_id,
            "title": self.title,
            "summary": self.summary,
            "keywords": self.keywords,
            "size": self.size,
            "members": self.members,
            "score": self.score,
        }


# ──────────────────────────────────────────────────────────────────────
# Top-level results
# ──────────────────────────────────────────────────────────────────────

@dataclass
class GraphBuildResult:
    """Returned by GraphBuilder.build_video_graph()."""
    video_id: str
    n_nodes: int
    n_edges: int
    n_communities: int
    elapsed_seconds: float
    sample_entities: List[str] = field(default_factory=list)
    sample_relations: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "video_id": self.video_id,
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_communities": self.n_communities,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "sample_entities": self.sample_entities,
            "sample_relations": self.sample_relations,
        }


@dataclass
class GraphSearchResult:
    """Returned by GraphSearch (local or global)."""
    mode: str                                 # "local" | "global" | "hybrid"
    seed_entities: List[str] = field(default_factory=list)
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    edges: List[Dict[str, Any]] = field(default_factory=list)
    chunk_ids: List[str] = field(default_factory=list)
    communities: List[Dict[str, Any]] = field(default_factory=list)
    context_text: str = ""                    # ready-to-feed-LLM context block

    def as_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "seed_entities": self.seed_entities,
            "n_nodes": len(self.nodes),
            "n_edges": len(self.edges),
            "n_chunks": len(self.chunk_ids),
            "n_communities": len(self.communities),
            "nodes": self.nodes,
            "edges": self.edges,
            "chunk_ids": self.chunk_ids,
            "communities": self.communities,
            "context_text": self.context_text,
        }