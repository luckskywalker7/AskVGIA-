# tests/test_agents.py
"""
Tests d'intégration — Phase 2 : Architecture Agentic
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Suites :
  1. TestConversationMemory  – mémoire multi-tours
  2. TestVideoSearchTools    – outils LangChain (sans LLM)
  3. TestAgentNodes          – nœuds LangGraph
  4. TestVideoSearchAgent    – agent complet (LLM Mock)
  5. TestMultiTurnChat       – conversation multi-tours

Pré-requis :
  - Phase 1 installée (storage, search_strategies, sparse_embeddings)
  - pip install langgraph langchain-core
  - LLM_PROVIDER=mock dans les settings (ou via env var)
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import unittest
from pathlib import Path

# ── Racine du projet dans le PYTHONPATH ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Forcer le mode mock pour les tests (pas de Ollama requis)
os.environ.setdefault("LLM_PROVIDER", "mock")

from agents.memory import ConversationMemory, ConversationTurn
from agents.nodes import should_continue
from agents.orchestrator import AgentResponse, VideoSearchAgent
from agents.state import AgentState, Citation
from agents.tools import VideoSearchTools
from core.search_strategies import HybridSearchEngine
from core.sparse_embeddings import SparseEmbeddingEngine
from core.storage import VideoStore

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)-8s] %(name)s – %(message)s"
)
logger = logging.getLogger("test_agents")

# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

DIM    = 768
RNG    = random.Random(42)
VID_ID = "test_agent_video_001"

CORPUS = [
    {"text": "Deep learning uses neural networks with multiple layers",
     "start_time": 30.0, "end_time": 60.0,
     "language": "en", "speaker": "SPEAKER_00", "topic": "Deep learning"},
    {"text": "Transformer models use self-attention mechanisms",
     "start_time": 60.0, "end_time": 90.0,
     "language": "en", "speaker": "SPEAKER_01", "topic": "Transformers"},
    {"text": "Generative AI can produce text, images and code",
     "start_time": 90.0, "end_time": 120.0,
     "language": "en", "speaker": "SPEAKER_00", "topic": "Generative AI"},
    {"text": "Reinforcement learning uses reward signals to train agents",
     "start_time": 120.0, "end_time": 150.0,
     "language": "en", "speaker": "SPEAKER_01", "topic": "RL"},
    {"text": "Computer vision enables image recognition and object detection",
     "start_time": 150.0, "end_time": 180.0,
     "language": "en", "speaker": "SPEAKER_00", "topic": "Computer vision"},
]

def rnd_vec(dim=DIM, seed=None):
    r = random.Random(seed)
    return [r.uniform(-1, 1) for _ in range(dim)]


class MockDenseEncoder:
    """Encodeur dense fictif (retourne un vecteur aléatoire fixe)."""
    def encode(self, text: str):
        return rnd_vec(seed=hash(text) % 10000)

    def encode_batch(self, texts):
        return [self.encode(t) for t in texts]


def _setup_store() -> tuple[VideoStore, HybridSearchEngine, SparseEmbeddingEngine]:
    """Crée un store prêt pour les tests avec données de test."""
    store  = VideoStore()
    sparse = SparseEmbeddingEngine()
    engine = HybridSearchEngine(store, sparse)

    store.reset()
    encoder     = MockDenseEncoder()
    dense_vecs  = [encoder.encode(s["text"]) for s in CORPUS]
    sparse_vecs = sparse.encode_batch([s["text"] for s in CORPUS])
    store.add_video(VID_ID, CORPUS, dense_vecs, sparse_vecs)

    return store, engine, sparse


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 1 – ConversationMemory
# ──────────────────────────────────────────────────────────────────────────────

class TestConversationMemory(unittest.TestCase):

    def setUp(self):
        self.mem = ConversationMemory(max_turns=3, session_id="test_session")

    def test_add_turn(self):
        turn = self.mem.add("question 1", "answer 1")
        self.assertEqual(self.mem.turn_count, 1)
        self.assertIsInstance(turn, ConversationTurn)

    def test_sliding_window(self):
        for i in range(5):
            self.mem.add(f"q{i}", f"a{i}")
        # max_turns=3 → seulement les 3 derniers
        self.assertEqual(self.mem.turn_count, 3)
        self.assertEqual(self.mem.get_last(1)[0].query, "q4")

    def test_context_prompt_empty(self):
        self.assertEqual(self.mem.get_context_prompt(), "")

    def test_context_prompt_with_turns(self):
        self.mem.add("What is deep learning?", "Deep learning is...")
        ctx = self.mem.get_context_prompt()
        self.assertIn("deep learning", ctx.lower())
        self.assertIn("Turn 1", ctx)

    def test_is_followup_detection(self):
        self.mem.add("What is AI?", "AI stands for...")
        self.assertTrue( self.mem.is_followup("and what about that?"))
        self.assertFalse(self.mem.is_followup("Explain quantum computing in detail"))

    def test_standalone_query_expansion(self):
        self.mem.add("What is deep learning?", "Deep learning is...")
        expanded = self.mem.build_standalone_query("and what about that?")
        self.assertIn("deep learning", expanded.lower())

    def test_clear(self):
        self.mem.add("q1", "a1")
        self.mem.clear()
        self.assertEqual(self.mem.turn_count, 0)

    def test_save_load(self, tmp_path=None):
        self.mem.add("q1", "a1", citations=[{"timestamp": "00:30"}])
        path = self.mem.save("test_session_save.json")
        self.assertTrue(Path(path).exists())

        mem2 = ConversationMemory(max_turns=5)
        n = mem2.load(str(path))
        self.assertEqual(n, 1)
        self.assertEqual(mem2.get_last(1)[0].query, "q1")

        Path(path).unlink(missing_ok=True)

    def test_stats(self):
        self.mem.add("q1", "a1", tool_calls=["search_video_segments"])
        stats = self.mem.stats()
        self.assertEqual(stats["turn_count"],       1)
        self.assertEqual(stats["total_tool_calls"],  1)


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 2 – VideoSearchTools
# ──────────────────────────────────────────────────────────────────────────────

class TestVideoSearchTools(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.store, cls.engine, _ = _setup_store()
        cls.encoder  = MockDenseEncoder()
        cls.factory  = VideoSearchTools(cls.store, cls.engine, cls.encoder)
        cls.tools    = cls.factory.build()
        cls.tool_map = {t.name: t for t in cls.tools}

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_video(VID_ID)

    def test_five_tools_built(self):
        self.assertEqual(len(self.tools), 5)
        expected = {
            "search_video_segments", "get_video_timeline",
            "get_segment_context",   "compare_speakers",
            "summarize_topic",
        }
        self.assertEqual({t.name for t in self.tools}, expected)

    def test_search_video_segments_returns_json(self):
        tool   = self.tool_map["search_video_segments"]
        result = tool.invoke({"query": "deep learning", "video_id": VID_ID})
        data   = json.loads(result)
        self.assertIn(data["status"], ("ok", "no_results"))
        if data["status"] == "ok":
            self.assertIn("results", data)
            self.assertIsInstance(data["results"], list)

    def test_get_video_timeline_returns_json(self):
        tool   = self.tool_map["get_video_timeline"]
        result = tool.invoke({"video_id": VID_ID})
        data   = json.loads(result)
        self.assertIn(data["status"], ("ok", "not_found"))
        if data["status"] == "ok":
            self.assertIn("timeline", data)
            self.assertGreater(data["total"], 0)

    def test_compare_speakers_returns_json(self):
        tool   = self.tool_map["compare_speakers"]
        result = tool.invoke({
            "speaker_a": "SPEAKER_00",
            "speaker_b": "SPEAKER_01",
            "topic":     "deep learning",
            "video_id":  VID_ID,
        })
        data = json.loads(result)
        self.assertIn(data["status"], ("ok", "error"))

    def test_summarize_topic_returns_json(self):
        tool   = self.tool_map["summarize_topic"]
        result = tool.invoke({"topic": "AI", "video_id": VID_ID})
        data   = json.loads(result)
        self.assertIn(data["status"], ("ok", "no_results"))

    def test_tools_handle_unknown_video(self):
        """Un video_id inexistant ne doit pas lever d'exception."""
        tool   = self.tool_map["search_video_segments"]
        result = tool.invoke({"query": "test", "video_id": "nonexistent_999"})
        data   = json.loads(result)
        self.assertIn("status", data)

    def test_search_all_modes(self):
        """Tester les 3 modes de recherche (dense/sparse/hybrid)."""
        tool = self.tool_map["search_video_segments"]
        for mode in ("dense", "sparse", "hybrid"):
            with self.subTest(mode=mode):
                result = tool.invoke({
                    "query": "transformers attention",
                    "video_id": VID_ID,
                    "mode": mode,
                })
                data = json.loads(result)
                self.assertIn("status", data)


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 3 – Nœuds LangGraph
# ──────────────────────────────────────────────────────────────────────────────

class TestAgentNodes(unittest.TestCase):

    def _make_state(self, last_msg) -> AgentState:
        return {
            "messages":       [last_msg],
            "query":          "test query",
            "video_id":       None,
            "search_context": None,
            "citations":      [],
            "final_answer":   "",
            "error":          None,
            "iteration":      0,
            "max_iterations": 8,
        }

    def test_should_continue_with_tool_calls(self):
        """AIMessage avec tool_calls → routage vers 'tools'."""
        msg = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "search_video_segments", "args": {}}]
        )
        state = self._make_state(msg)
        self.assertEqual(should_continue(state), "tools")

    def test_should_continue_without_tool_calls(self):
        """AIMessage sans tool_calls → routage vers 'end'."""
        from langchain_core.messages import AIMessage
        msg   = AIMessage(content="Here is my final answer.")
        state = self._make_state(msg)
        self.assertEqual(should_continue(state), "end")

    def test_should_continue_max_iterations(self):
        """Limite d'itérations atteinte → 'end' forcé."""
        from langchain_core.messages import AIMessage
        msg   = AIMessage(
            content="",
            tool_calls=[{"id": "c1", "name": "tool", "args": {}}]
        )
        state = self._make_state(msg)
        state["iteration"]      = 10
        state["max_iterations"] = 8
        self.assertEqual(should_continue(state), "end")


# Import ici pour éviter les erreurs si langchain_core non installé
try:
    from langchain_core.messages import AIMessage
except ImportError:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 4 – VideoSearchAgent complet (Mock LLM)
# ──────────────────────────────────────────────────────────────────────────────

class TestVideoSearchAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.store, cls.engine, _ = _setup_store()
        cls.encoder = MockDenseEncoder()
        os.environ["LLM_PROVIDER"] = "mock"
        cls.agent = VideoSearchAgent(
            store=         cls.store,
            search_engine= cls.engine,
            dense_encoder= cls.encoder,
            session_id=    "test_agent_session",
        )

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_video(VID_ID)

    def test_agent_created(self):
        self.assertIsNotNone(self.agent)
        self.assertEqual(len(self.agent.tools), 5)

    def test_chat_returns_agent_response(self):
        resp = self.agent.chat("What is discussed in the video?")
        self.assertIsInstance(resp, AgentResponse)
        self.assertIsInstance(resp.answer, str)
        self.assertGreater(len(resp.answer), 0)

    def test_chat_with_video_id(self):
        resp = self.agent.chat(
            "Explain deep learning", video_id=VID_ID
        )
        self.assertIsInstance(resp, AgentResponse)
        self.assertEqual(resp.video_id, VID_ID)

    def test_chat_uses_tools(self):
        resp = self.agent.chat("Search for transformer content")
        # Le Mock LLM appelle search_video_segments
        self.assertIn("search_video_segments", resp.tool_calls)

    def test_memory_updated_after_chat(self):
        before = self.agent.memory.turn_count
        self.agent.chat("What about reinforcement learning?")
        self.assertEqual(self.agent.memory.turn_count, before + 1)

    def test_reset_clears_memory(self):
        self.agent.chat("test question")
        self.agent.reset()
        self.assertEqual(self.agent.memory.turn_count, 0)

    def test_error_handling(self):
        """Le chat doit retourner une réponse même si le LLM échoue."""
        resp = self.agent.chat("")   # requête vide
        self.assertIsInstance(resp, AgentResponse)

    def test_repr(self):
        r = repr(self.agent)
        self.assertIn("VideoSearchAgent", r)
        self.assertIn("mock", r.lower())


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 5 – Conversation multi-tours
# ──────────────────────────────────────────────────────────────────────────────

class TestMultiTurnChat(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.store, cls.engine, _ = _setup_store()
        cls.encoder = MockDenseEncoder()
        os.environ["LLM_PROVIDER"] = "mock"
        cls.agent = VideoSearchAgent(
            store=         cls.store,
            search_engine= cls.engine,
            dense_encoder= cls.encoder,
            session_id=    "multi_turn_test",
        )

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_video(VID_ID)

    def test_three_turns(self):
        """3 tours successifs sans erreur."""
        queries = [
            "What topics are covered in the video?",
            "Tell me more about transformers",
            "And what about reinforcement learning?",
        ]
        for i, q in enumerate(queries, 1):
            with self.subTest(turn=i):
                resp = self.agent.chat(q, video_id=VID_ID)
                self.assertIsInstance(resp, AgentResponse)
                self.assertGreater(len(resp.answer), 0)

        self.assertEqual(self.agent.memory.turn_count, 3)

    def test_followup_query_expansion(self):
        """Un follow-up doit être résolu avec le contexte précédent."""
        self.agent.reset()
        self.agent.chat("What is deep learning?")
        expanded = self.agent.memory.build_standalone_query("and that?")
        self.assertIn("deep learning", expanded.lower())

    def test_session_stats(self):
        """Les statistiques de session doivent être cohérentes."""
        stats = self.agent.memory.stats()
        self.assertIn("turn_count",       stats)
        self.assertIn("total_tool_calls", stats)
        self.assertIn("session_id",       stats)
        self.assertGreater(stats["turn_count"], 0)


# ──────────────────────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  TEST SUITE — Phase 2 : Architecture Agentic (LangGraph)")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestConversationMemory,
        TestVideoSearchTools,
        TestAgentNodes,
        TestVideoSearchAgent,
        TestMultiTurnChat,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"  Réussis  : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Échecs   : {len(result.failures)}")
    print(f"  Erreurs  : {len(result.errors)}")
    print("=" * 70)
    sys.exit(0 if result.wasSuccessful() else 1)
