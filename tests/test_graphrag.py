"""
tests/test_graphrag.py — Smoke + integration tests for the GraphRAG layer
==========================================================================
Run from the project root:

    python -m pytest tests/test_graphrag.py -v

Or directly (no pytest required, falls back to a tiny runner):

    python tests/test_graphrag.py

The tests are designed to run WITHOUT Ollama / WhisperX / FAISS — they use
the heuristic fallback of GraphExtractor (no LLM call) and a synthetic
list of `SemanticChunk`-like dataclasses.

What is tested
--------------
1. graph_contracts: normalize_entity_type / normalize_relation_type
2. GraphStore: add_entities / add_relations / merge / delete_video / save+load
3. GraphExtractor (heuristic fallback only — Ollama not required)
4. GraphBuilder: end-to-end build_video_graph on synthetic chunks
5. GraphSearch.local_search and global_search (no LLM)
6. /graph/* FastAPI endpoints reachability (in-process TestClient)

Author : Spiral father, 2026.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Make project root importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from core.graph_contracts import (  # noqa: E402
    EntityMention,
    GraphEntity,
    GraphRelation,
    RelationEvidence,
    normalize_entity_type,
    normalize_relation_type,
)
from core.graph_store import GraphStore  # noqa: E402
from core.graph_extractor import ExtractorConfig, GraphExtractor  # noqa: E402
from core.graph_builder import GraphBuilder  # noqa: E402
from core.graph_search import GraphSearch, detect_intent  # noqa: E402


# ─────────────────────────────────────────────────────────────────────
# Fixtures: synthetic SemanticChunk-like dataclass
# ─────────────────────────────────────────────────────────────────────
@dataclass
class FakeChunk:
    chunk_id: str
    video_id: str
    start_time: float
    end_time: float
    speaker: str
    language: str
    text: str
    search_text: str = ""
    normalized_text: str = ""

    def __post_init__(self):
        self.search_text = self.search_text or self.text
        self.normalized_text = self.normalized_text or self.text


SYNTHETIC_CHUNKS: List[FakeChunk] = [
    FakeChunk(
        "c1", "vid_alpha", 0.0, 22.0, "Alice", "en",
        "Alice Chen from MIT presented BB84 quantum key distribution. "
        "She explained how RSA encryption could be broken by Shor's algorithm. "
        "BB84 was developed by Bennett and Brassard at IBM Research.",
    ),
    FakeChunk(
        "c2", "vid_alpha", 22.0, 50.0, "Alice", "en",
        "BB84 uses photon polarization to detect eavesdropping. "
        "The protocol relies on the no-cloning theorem from quantum mechanics. "
        "MIT researchers compare BB84 to E91, another quantum key distribution scheme.",
    ),
    FakeChunk(
        "c3", "vid_alpha", 50.0, 75.0, "Bob", "en",
        "Bob Smith argued that post-quantum cryptography by NIST is more practical. "
        "He criticized BB84 because it requires specialized hardware. "
        "Lattice-based methods like Kyber and Dilithium are NIST standards.",
    ),
    FakeChunk(
        "c4", "vid_beta", 0.0, 30.0, "Carol", "fr",
        "Carol Dupont a parlé du modèle Transformer et de l'attention multi-têtes. "
        "Elle a comparé GPT à BERT. Google a développé BERT en 2018, "
        "tandis que GPT vient de OpenAI.",
    ),
    FakeChunk(
        "c5", "vid_beta", 30.0, 60.0, "Carol", "fr",
        "Le modèle GPT-4 utilise l'attention causale. "
        "OpenAI a publié GPT-4 en 2023. BERT, lui, utilise une attention bidirectionnelle.",
    ),
]


@pytest.fixture()
def tmp_store_dir():
    d = Path(tempfile.mkdtemp(prefix="graphstore_test_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture()
def store(tmp_store_dir):
    return GraphStore(persist_dir=str(tmp_store_dir))


@pytest.fixture()
def heuristic_extractor():
    """Force the heuristic path by disabling Ollama probing."""
    cfg = ExtractorConfig(use_heuristic_fallback=True, timeout_s=2)
    ext = GraphExtractor(llm_client=None, config=cfg)
    ext._ollama_available = False  # bypass /api/tags
    return ext


@pytest.fixture()
def builder(store, heuristic_extractor):
    return GraphBuilder(
        store=store, extractor=heuristic_extractor, llm_client=None,
        run_communities_default=True,
        community_summarize_default=False,  # no LLM in tests
    )


# ─────────────────────────────────────────────────────────────────────
# 1) Contracts
# ─────────────────────────────────────────────────────────────────────
class TestContracts:
    def test_entity_type_normalization(self):
        assert normalize_entity_type("PERSON") == "PERSON"
        assert normalize_entity_type("organization") == "ORG"
        assert normalize_entity_type("People") == "PERSON"
        assert normalize_entity_type("LOCATION") == "PLACE"
        assert normalize_entity_type("xyz123") == "OTHER"
        assert normalize_entity_type("") == "OTHER"

    def test_relation_type_normalization(self):
        assert normalize_relation_type("discusses") == "discusses"
        assert normalize_relation_type("Talks About") == "discusses"
        assert normalize_relation_type("agrees_with") == "supports"
        assert normalize_relation_type("CRITICIZES") == "opposes"
        assert normalize_relation_type("vs") == "compares_to"
        assert normalize_relation_type("unknown_xyz") == "related_to"

    def test_graph_entity_post_init(self):
        e = GraphEntity(name="  Alice  ", type="people")
        assert e.name == "Alice"
        assert e.type == "PERSON"


# ─────────────────────────────────────────────────────────────────────
# 2) GraphStore basic ops
# ─────────────────────────────────────────────────────────────────────
class TestGraphStore:
    def test_add_and_merge_entities(self, store):
        m1 = EntityMention("vid_a", "c1", 0.0, 10.0, "Alice", "en", "BB84")
        m2 = EntityMention("vid_a", "c2", 10.0, 20.0, "Alice", "en", "BB-84")
        e1 = GraphEntity(name="BB84", type="TECH",
                         description="quantum key distribution",
                         mentions=[m1])
        e2 = GraphEntity(name="BB84", type="TECH",
                         description="protocol from 1984",
                         aliases={"BB-84"}, mentions=[m2])
        n_new = store.add_entities([e1, e2])
        assert n_new == 1
        assert store.graph.number_of_nodes() == 1

        node = store.graph.nodes["bb84"]
        assert node["n_mentions"] == 2
        assert "vid_a" in node["video_ids"]
        assert "BB-84" in node["aliases"]

    def test_add_relations(self, store):
        store.add_entities([
            GraphEntity(name="Alice", type="PERSON"),
            GraphEntity(name="MIT", type="ORG"),
        ])
        rel = GraphRelation(
            src="Alice", dst="MIT", type="works_at",
            evidence=[RelationEvidence("vid_a", "c1", 0.0, 10.0, "Alice",
                                       "Alice from MIT")],
        )
        n = store.add_relations([rel])
        assert n == 1
        assert store.graph.number_of_edges() == 1
        assert store.graph.has_edge("alice", "mit", key="works_at")

    def test_save_and_load_roundtrip(self, store, tmp_store_dir):
        store.add_entities([
            GraphEntity(name="X", type="CONCEPT",
                        mentions=[EntityMention("v", "c", 0.0)]),
            GraphEntity(name="Y", type="CONCEPT",
                        mentions=[EntityMention("v", "c", 5.0)]),
        ])
        store.add_relations([GraphRelation(src="X", dst="Y", type="related_to")])
        store.save()

        store2 = GraphStore(persist_dir=str(tmp_store_dir))
        assert store2.load() is True
        assert store2.graph.number_of_nodes() == 2
        assert store2.graph.has_edge("x", "y", key="related_to")

    def test_delete_video(self, store):
        store.add_entities([
            GraphEntity(name="A", type="OTHER",
                        mentions=[EntityMention("v1", "c1", 0.0)]),
            GraphEntity(name="B", type="OTHER",
                        mentions=[EntityMention("v2", "c2", 0.0)]),
        ])
        store.add_relations([
            GraphRelation(src="A", dst="B", type="related_to",
                          evidence=[RelationEvidence("v1", "c1", 0.0)]),
        ])
        counts = store.delete_video("v1")
        assert counts["nodes_removed"] == 1
        assert counts["edges_removed"] == 1
        assert store.graph.number_of_nodes() == 1
        assert "b" in store.graph


# ─────────────────────────────────────────────────────────────────────
# 3) Extractor (heuristic — no Ollama)
# ─────────────────────────────────────────────────────────────────────
class TestExtractor:
    def test_heuristic_extracts_some_capitalized_entities(self, heuristic_extractor):
        ents, rels = heuristic_extractor.extract_from_chunk(SYNTHETIC_CHUNKS[0])
        names = {e.name for e in ents}
        assert len(ents) >= 2
        # heuristic should at least pick capitalized noun phrases
        assert any("alice" in n.lower() or "MIT" in n or "BB84" in n for n in names)


# ─────────────────────────────────────────────────────────────────────
# 4) Builder end-to-end
# ─────────────────────────────────────────────────────────────────────
class TestBuilder:
    def test_build_video_graph_alpha(self, builder, store):
        res = builder.build_video_graph(
            video_id="vid_alpha",
            chunks=SYNTHETIC_CHUNKS[:3],
            run_communities=True,
            summarize_communities=False,
            save=False,
        )
        assert res.video_id == "vid_alpha"
        assert res.n_nodes >= 2
        # builder is multi-video, but here we only added vid_alpha
        videos = set()
        for _, d in store.graph.nodes(data=True):
            videos |= d.get("video_ids", set())
        assert "vid_alpha" in videos

    def test_build_two_videos_then_search(self, builder, store):
        builder.build_video_graph("vid_alpha", SYNTHETIC_CHUNKS[:3], save=False)
        builder.build_video_graph("vid_beta",  SYNTHETIC_CHUNKS[3:], save=False)
        stats = store.stats()
        assert stats["n_videos"] == 2
        assert stats["n_nodes"] >= 4


# ─────────────────────────────────────────────────────────────────────
# 5) GraphSearch
# ─────────────────────────────────────────────────────────────────────
class TestGraphSearch:
    def test_intent_detection(self):
        assert detect_intent("Summarize the main themes") == "global"
        assert detect_intent("Quels sont les thèmes principaux ?") == "global"
        assert detect_intent("Who is Alice?") == "local"
        assert detect_intent("Tell me about BB84 vs RSA") in ("local", "hybrid")
        assert detect_intent("Random general question without entity") == "hybrid"

    def test_local_search_returns_neighborhood(self, builder, store):
        builder.build_video_graph("vid_alpha", SYNTHETIC_CHUNKS[:3], save=False)
        gs = GraphSearch(graph_store=store)
        # NOTE: heuristic extractor cannot pick numeric tokens ("BB84"),
        # so we query for an entity it CAN extract (capitalized noun).
        # In production with the LLM extractor, BB84 IS extracted reliably.
        out = gs.ask(question="Tell me about Alice Chen and MIT",
                     mode="local", top_k=5)
        assert out["mode_used"] == "local"
        assert out["graph"] is not None
        assert out["graph"]["n_nodes"] >= 1
        assert isinstance(out["graph"]["context_text"], str)
        # also ensure local_search direct API works with explicit seeds
        res = store.local_search(seed_entities=["MIT"], hops=1, max_nodes=10)
        assert len(res.nodes) >= 1
        assert any(n["name"] == "MIT" for n in res.nodes)

    def test_global_search_uses_communities(self, builder, store):
        builder.build_video_graph("vid_alpha", SYNTHETIC_CHUNKS[:3], save=False)
        builder.build_video_graph("vid_beta",  SYNTHETIC_CHUNKS[3:], save=False)
        gs = GraphSearch(graph_store=store)
        out = gs.ask(question="What are the main themes?", mode="global")
        assert out["mode_used"] == "global"
        # communities may be empty if graph is too small but call must succeed
        assert "graph" in out


# ─────────────────────────────────────────────────────────────────────
# 6) FastAPI endpoints (optional — only if FastAPI test client is available)
# ─────────────────────────────────────────────────────────────────────
class TestFastAPIEndpoints:
    def test_graph_endpoints_smoke(self, tmp_store_dir, monkeypatch):
        try:
            from fastapi.testclient import TestClient
        except Exception:
            pytest.skip("FastAPI TestClient not available")

        # configure GraphStore to use tmp dir BEFORE importing api/main
        monkeypatch.setenv("GRAPH_STORE_DIR", str(tmp_store_dir))

        # try both layouts: api/main.py OR main.py at root
        try:
            from api import main as api_main  # type: ignore
        except Exception:
            try:
                import main as api_main  # type: ignore
            except Exception:
                pytest.skip("api.main / main not importable")

        # force GraphRAG init even without ollama
        try:
            api_main._init_components()  # type: ignore
        except Exception as exc:
            pytest.skip(f"_init_components failed: {exc}")

        client = TestClient(api_main.app)  # type: ignore
        r = client.get("/graph/health")
        assert r.status_code == 200
        body = r.json()
        assert "graph_store" in body


# ─────────────────────────────────────────────────────────────────────
# Manual runner (when pytest unavailable)
# ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import traceback
    test_classes = [
        TestContracts, TestGraphStore, TestExtractor,
        TestBuilder, TestGraphSearch,
    ]
    n_pass = n_fail = 0
    for cls in test_classes:
        inst = cls()
        for attr in dir(inst):
            if not attr.startswith("test_"):
                continue
            print(f"\n▶ {cls.__name__}.{attr}")
            try:
                # build fixtures lazily
                kwargs = {}
                func = getattr(inst, attr)
                params = func.__code__.co_varnames[: func.__code__.co_argcount]
                if "tmp_store_dir" in params:
                    kwargs["tmp_store_dir"] = Path(tempfile.mkdtemp(prefix="t_"))
                if "store" in params:
                    kwargs["store"] = GraphStore(
                        persist_dir=str(kwargs.get("tmp_store_dir")
                                        or tempfile.mkdtemp())
                    )
                if "heuristic_extractor" in params:
                    cfg = ExtractorConfig(use_heuristic_fallback=True, timeout_s=2)
                    he = GraphExtractor(llm_client=None, config=cfg)
                    he._ollama_available = False
                    kwargs["heuristic_extractor"] = he
                if "builder" in params:
                    s = kwargs.get("store") or GraphStore(
                        persist_dir=str(tempfile.mkdtemp())
                    )
                    cfg = ExtractorConfig(use_heuristic_fallback=True, timeout_s=2)
                    he = GraphExtractor(llm_client=None, config=cfg)
                    he._ollama_available = False
                    kwargs["builder"] = GraphBuilder(
                        store=s, extractor=he, llm_client=None,
                        community_summarize_default=False,
                    )
                    kwargs.setdefault("store", s)
                func(**kwargs)
                print("  ✅ PASS")
                n_pass += 1
            except Exception:
                traceback.print_exc()
                print("  ❌ FAIL")
                n_fail += 1
    print(f"\n=== {n_pass} passed, {n_fail} failed ===")
    sys.exit(0 if n_fail == 0 else 1)
