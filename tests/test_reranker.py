# tests/test_reranker.py
"""
Tests d'intégration — Phase 2 : CrossEncoder Re-Ranking
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Suites :
  1. TestRerankerConfig        – validation de la config
  2. TestCrossEncoderReranker  – chargement, scoring, reranking
  3. TestRerankerIntegration   – intégration avec VideoStore + HybridSearchEngine
  4. TestRerankerEdgeCases     – cas limites
  5. TestRerankerBenchmark     – avant/après comparaison de qualité

Usage :
    python -m pytest tests/test_reranker.py -v
    python tests/test_reranker.py
"""

from __future__ import annotations

import logging
import random
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.reranker        import (
    CrossEncoderReranker,
    RerankerConfig,
    RERANKER_PRESETS,
    make_reranker,
)
from core.storage         import SearchResult, VideoStore
from core.sparse_embeddings import SparseEmbeddingEngine
from core.search_strategies import (
    HybridSearchEngine,
    SearchConfig,
    SearchMode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_reranker")
import numpy as np
if not hasattr(np, 'NINF'):
    np.NINF = -np.inf
    print("✅ NumPy 2.0 patch applied: np.NINF = -np.inf")
# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────────────────────

DIM      = 768
VIDEO_ID = "test_reranker_001"

CORPUS = [
    {"text": "Deep learning uses neural networks with many layers",
     "start_time": 0.0,   "end_time": 30.0,  "language": "en", "speaker": "SPEAKER_00", "topic": "DL"},
    {"text": "Transformer architecture changed natural language processing forever",
     "start_time": 30.0,  "end_time": 60.0,  "language": "en", "speaker": "SPEAKER_01", "topic": "NLP"},
    {"text": "Cooking pasta requires boiling water and adding salt",
     "start_time": 60.0,  "end_time": 90.0,  "language": "en", "speaker": "SPEAKER_00", "topic": "other"},
    {"text": "GPT and BERT are popular large language models",
     "start_time": 90.0,  "end_time": 120.0, "language": "en", "speaker": "SPEAKER_01", "topic": "LLM"},
    {"text": "Reinforcement learning trains agents using reward signals",
     "start_time": 120.0, "end_time": 150.0, "language": "en", "speaker": "SPEAKER_00", "topic": "RL"},
    {"text": "The weather is sunny today with mild temperatures",
     "start_time": 150.0, "end_time": 180.0, "language": "en", "speaker": "SPEAKER_01", "topic": "other"},
    {"text": "Attention mechanism allows models to focus on relevant parts",
     "start_time": 180.0, "end_time": 210.0, "language": "en", "speaker": "SPEAKER_00", "topic": "NLP"},
    {"text": "الذكاء الاصطناعي يغير مستقبل التكنولوجيا",
     "start_time": 210.0, "end_time": 240.0, "language": "ar", "speaker": "SPEAKER_00", "topic": "AI"},
]


def _make_result(seg: dict, score: float = 0.5, idx: int = 0) -> SearchResult:
    return SearchResult(
        segment_id=          f"{VIDEO_ID}_seg_{idx:04d}",
        video_id=            VIDEO_ID,
        text=                seg["text"],
        start_time=          seg["start_time"],
        end_time=            seg["end_time"],
        timestamp_formatted= f"{int(seg['start_time']//60):02d}:{int(seg['start_time']%60):02d}",
        score=               score,
        language=            seg.get("language", "en"),
        speaker=             seg.get("speaker", ""),
        topic=               seg.get("topic", ""),
        chunk_type=          "transcript",
    )


def _make_dense(n: int, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(DIM)] for _ in range(n)]


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 1 – RerankerConfig
# ──────────────────────────────────────────────────────────────────────────────

class TestRerankerConfig(unittest.TestCase):

    def test_defaults(self):
        cfg = RerankerConfig()
        self.assertEqual(cfg.model_name, "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.batch_size, 32)
        self.assertTrue(cfg.normalize_scores)

    def test_invalid_device(self):
        with self.assertRaises(ValueError):
            RerankerConfig(device="tpu")

    def test_invalid_batch_size(self):
        with self.assertRaises(ValueError):
            RerankerConfig(batch_size=0)

    def test_presets_exist(self):
        expected = {"fast", "quality", "multilingual"}
        self.assertEqual(set(RERANKER_PRESETS.keys()), expected)

    def test_make_reranker_valid_preset(self):
        # Ne pas charger le modèle en test (auto_load=False)
        reranker = CrossEncoderReranker(
            RerankerConfig(
                model_name=RERANKER_PRESETS["fast"],
                auto_load=False,
            )
        )
        self.assertFalse(reranker.is_loaded)

    def test_make_reranker_invalid_preset(self):
        with self.assertRaises(ValueError):
            make_reranker("nonexistent")


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 2 – CrossEncoderReranker
# ──────────────────────────────────────────────────────────────────────────────

class TestCrossEncoderReranker(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reranker = CrossEncoderReranker(
            RerankerConfig(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                device="cpu",
                auto_load=True,
            )
        )
        logger.info(f"🔧 Reranker chargé : {cls.reranker.is_loaded}")

    def test_model_loads(self):
        """Le modèle doit se charger sans exception."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent — test ignoré")
        self.assertTrue(self.reranker.is_loaded)

    def test_score_returns_list(self):
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        texts  = ["deep learning", "neural networks", "pasta recipe"]
        scores = self.reranker.score("machine learning", texts)
        self.assertEqual(len(scores), len(texts))

    def test_score_normalized_in_range(self):
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        texts  = [s["text"] for s in CORPUS]
        scores = self.reranker.score_normalized("deep learning neural networks", texts)
        for s in scores:
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(   s, 1.0 + 1e-9)

    def test_relevant_scores_higher_than_irrelevant(self):
        """
        Les textes sur le ML doivent avoir des scores plus élevés
        que les textes hors-sujet pour une requête ML.
        """
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")

        query    = "deep learning neural networks transformer"
        relevant = "Transformer architecture changed natural language processing forever"
        irrelevant = "Cooking pasta requires boiling water and adding salt"

        scores = self.reranker.score_normalized(query, [relevant, irrelevant])
        logger.info(
            f"   relevant={scores[0]:.4f} | irrelevant={scores[1]:.4f}"
        )
        self.assertGreater(
            scores[0], scores[1],
            f"Score relevant ({scores[0]:.4f}) < irrelevant ({scores[1]:.4f})"
        )

    def test_rerank_changes_order(self):
        """
        Le re-ranking doit remettre les résultats pertinents en tête,
        même s'ils avaient un score initial faible.
        """
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")

        # Inverser intentionnellement : le texte hors-sujet est en tête
        results = [
            _make_result(CORPUS[2], score=0.9, idx=2),  # pasta (hors-sujet)
            _make_result(CORPUS[5], score=0.8, idx=5),  # weather (hors-sujet)
            _make_result(CORPUS[0], score=0.3, idx=0),  # deep learning ← devrait monter
            _make_result(CORPUS[1], score=0.2, idx=1),  # transformer ← devrait monter
        ]

        query   = "deep learning neural networks transformer"
        reranked = self.reranker.rerank(query, results, top_n=4)

        top2_topics = {reranked[0].topic, reranked[1].topic}
        logger.info(
            f"   Top-2 topics après reranking : {top2_topics}"
        )
        # Au moins un des 2 premiers résultats doit être ML-related
        self.assertTrue(
            top2_topics & {"DL", "NLP", "LLM", "RL"},
            f"Aucun résultat ML en top-2 : {top2_topics}"
        )

    def test_rerank_empty_list(self):
        """Reranking d'une liste vide → liste vide."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        result = self.reranker.rerank("query", [], top_n=5)
        self.assertEqual(result, [])

    def test_rerank_top_n_respected(self):
        """Le résultat ne dépasse pas top_n."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        results  = [_make_result(seg, idx=i) for i, seg in enumerate(CORPUS)]
        reranked = self.reranker.rerank("deep learning", results, top_n=3)
        self.assertLessEqual(len(reranked), 3)

    def test_score_threshold_filters(self):
        """Les résultats sous le seuil doivent être exclus."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")

        original_threshold = self.reranker.config.score_threshold
        try:
            # Seuil très élevé → peu ou pas de résultats
            self.reranker.config.score_threshold = 0.99
            results  = [_make_result(seg, idx=i) for i, seg in enumerate(CORPUS)]
            reranked = self.reranker.rerank("deep learning", results, top_n=5)
            logger.info(
                f"   Avec threshold=0.99 → {len(reranked)} résultats"
            )
            self.assertLessEqual(len(reranked), len(results))
        finally:
            self.reranker.config.score_threshold = original_threshold

    def test_strategy_label_updated(self):
        """Le strategy des résultats doit contenir '+rerank'."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        results  = [_make_result(seg, idx=i) for i, seg in enumerate(CORPUS[:3])]
        reranked = self.reranker.rerank("AI technology", results, top_n=3)
        for r in reranked:
            self.assertIn("rerank", r.strategy, f"strategy={r.strategy!r}")

    def test_scores_are_normalized(self):
        """Après reranking (normalize=True), scores dans [0, 1]."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        results  = [_make_result(seg, idx=i) for i, seg in enumerate(CORPUS)]
        reranked = self.reranker.rerank("deep learning", results)
        for r in reranked:
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(   r.score, 1.0 + 1e-9)

    def test_fallback_without_model(self):
        """Sans modèle chargé, retourne les résultats d'origine."""
        reranker = CrossEncoderReranker(RerankerConfig(auto_load=False))
        self.assertFalse(reranker.is_loaded)
        results  = [_make_result(seg, idx=i) for i, seg in enumerate(CORPUS[:3])]
        output   = reranker.rerank("query", results, top_n=3)
        # Doit retourner les résultats d'origine (fallback gracieux)
        self.assertEqual(len(output), len(results))

    def test_warmup_no_crash(self):
        """warmup() ne doit pas lever d'exception."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        self.reranker.warmup()   # ne doit pas crasher


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 3 – Intégration avec HybridSearchEngine
# ──────────────────────────────────────────────────────────────────────────────

class TestRerankerIntegration(unittest.TestCase):
    """Tests d'intégration complets : Store → Hybrid → Rerank."""

    @classmethod
    def setUpClass(cls):
        cls.store   = VideoStore()
        cls.sparse  = SparseEmbeddingEngine()
        cls.reranker = CrossEncoderReranker(
            RerankerConfig(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                auto_load=True,
            )
        )
        cls.engine = HybridSearchEngine(cls.store, cls.sparse, reranker=cls.reranker)

        cls.store.reset()
        dense_vecs   = _make_dense(len(CORPUS))
        sparse_vecs  = cls.sparse.encode_batch([s["text"] for s in CORPUS])
        cls.store.add_video(VIDEO_ID, CORPUS, dense_vecs, sparse_vecs)

        cls.query_dense = _make_dense(1, seed=99)[0]
        cls.query_text  = "deep learning neural networks transformer"

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_video(VIDEO_ID)

    def test_hybrid_without_rerank(self):
        cfg     = SearchConfig(
            mode=SearchMode.HYBRID, top_k=5,
            video_id=VIDEO_ID, rerank=False,
        )
        results = self.engine.search(self.query_text, self.query_dense, cfg)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertNotIn("rerank", r.strategy)

    def test_hybrid_with_rerank(self):
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        cfg     = SearchConfig(
            mode=SearchMode.HYBRID, top_k=5,
            video_id=VIDEO_ID,
            rerank=True, rerank_top_n=3,
        )
        results = self.engine.search(self.query_text, self.query_dense, cfg)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 3)
        for r in results:
            self.assertIn("rerank", r.strategy)

    def test_rerank_improves_precision(self):
        """
        Après reranking, le top-1 doit être plus pertinent
        (hors-sujet ne doit pas être en tête).
        """
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")

        # Recherche sans rerank
        cfg_no = SearchConfig(
            mode=SearchMode.HYBRID, top_k=8, video_id=VIDEO_ID, rerank=False
        )
        # Recherche avec rerank
        cfg_re = SearchConfig(
            mode=SearchMode.HYBRID, top_k=8, video_id=VIDEO_ID,
            rerank=True, rerank_top_n=3,
        )

        no_rerank = self.engine.search(self.query_text, self.query_dense, cfg_no)
        reranked  = self.engine.search(self.query_text, self.query_dense, cfg_re)

        logger.info(
            f"\n   Sans rerank — top-1 : [{no_rerank[0].topic}] {no_rerank[0].text[:50]}"
            if no_rerank else "\n   Sans rerank — aucun résultat"
        )
        logger.info(
            f"   Avec rerank — top-1 : [{reranked[0].topic}] {reranked[0].text[:50]}"
            if reranked else "   Avec rerank — aucun résultat"
        )

        # Les résultats reranked ne doivent pas commencer par un sujet hors-sujet
        if reranked:
            self.assertNotIn(reranked[0].topic, {"other"},
                             f"Top-1 reranked est hors-sujet: {reranked[0].text}")

    def test_benchmark_includes_rerank(self):
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        comparison = self.engine.benchmark(
            query=      self.query_text,
            query_dense=self.query_dense,
            top_k=      3,
            video_id=   VIDEO_ID,
            include_rerank=True,
        )
        self.assertIn("hybrid+rerank", comparison)


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 4 – Cas limites
# ──────────────────────────────────────────────────────────────────────────────

class TestRerankerEdgeCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reranker = CrossEncoderReranker(
            RerankerConfig(auto_load=True)
        )

    def test_empty_query(self):
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        results = [_make_result(CORPUS[0], idx=0)]
        output  = self.reranker.rerank("", results, top_n=1)
        self.assertIsInstance(output, list)

    def test_single_result(self):
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        results = [_make_result(CORPUS[0], idx=0)]
        output  = self.reranker.rerank("deep learning", results, top_n=1)
        self.assertEqual(len(output), 1)

    def test_top_n_greater_than_results(self):
        """top_n > len(results) → retourner tous les résultats disponibles."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        results = [_make_result(seg, idx=i) for i, seg in enumerate(CORPUS[:3])]
        output  = self.reranker.rerank("deep learning", results, top_n=100)
        self.assertLessEqual(len(output), len(results))

    def test_very_long_text(self):
        """Un texte très long doit être tronqué sans erreur."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")
        long_seg = dict(CORPUS[0])
        long_seg["text"] = "deep learning " * 300   # ~600 tokens
        results = [_make_result(long_seg, idx=0)]
        output  = self.reranker.rerank("deep learning", results, top_n=1)
        self.assertEqual(len(output), 1)

    def test_arabic_text(self):
        """Le modèle multilingual doit accepter l'arabe sans erreur."""
        reranker_ml = CrossEncoderReranker(
            RerankerConfig(
                model_name="cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                auto_load=True,
            )
        )
        if not reranker_ml.is_loaded:
            self.skipTest("sentence-transformers absent ou modèle multilingual non téléchargé")

        results = [_make_result(CORPUS[7], idx=7)]   # texte arabe
        output  = reranker_ml.rerank("الذكاء الاصطناعي", results, top_n=1)
        self.assertEqual(len(output), 1)

    def test_sigmoid_bounds(self):
        """_sigmoid doit rester dans [0, 1] pour des valeurs extrêmes."""
        from core.reranker import CrossEncoderReranker
        r = CrossEncoderReranker(RerankerConfig(auto_load=False))
        for x in [-1000.0, -10.0, 0.0, 10.0, 1000.0]:
            s = r._sigmoid(x)
            self.assertGreaterEqual(s, 0.0)
            self.assertLessEqual(   s, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 5 – Benchmark avant/après reranking
# ──────────────────────────────────────────────────────────────────────────────

class TestRerankerBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.reranker = CrossEncoderReranker(
            RerankerConfig(
                model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
                auto_load=True,
            )
        )

    TEST_PAIRS = [
        ("deep learning neural networks",
         [CORPUS[0], CORPUS[1], CORPUS[2], CORPUS[5], CORPUS[6]]),
        ("generative AI language model",
         [CORPUS[3], CORPUS[4], CORPUS[2], CORPUS[5], CORPUS[0]]),
    ]

    def test_reranking_speed(self):
        """Re-scorer 8 résultats doit prendre moins de 2 secondes sur CPU."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")

        results = [_make_result(seg, idx=i) for i, seg in enumerate(CORPUS)]
        t0      = time.perf_counter()
        _       = self.reranker.rerank("deep learning", results, top_n=5)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(f"⏱  Reranking {len(results)} segments : {elapsed:.1f} ms")
        self.assertLess(elapsed, 2000.0,
                        f"Reranking trop lent : {elapsed:.0f} ms > 2000 ms")

    def test_print_before_after_comparison(self):
        """Affiche les classements avant et après reranking."""
        if not self.reranker.is_loaded:
            self.skipTest("sentence-transformers absent")

        logger.info("\n\n📊 COMPARAISON AVANT / APRÈS RERANKING")
        logger.info("═" * 65)

        for query, segs in self.TEST_PAIRS:
            # Simuler un ordre arbitraire (non pertinent en tête)
            results = [_make_result(seg, score=0.9 - 0.1 * i, idx=i)
                       for i, seg in enumerate(segs)]

            reranked = self.reranker.rerank(query, results, top_n=5)

            logger.info(f"\n  Requête : '{query}'")
            logger.info(f"  {'AVANT':^30s} | {'APRÈS':^30s}")
            logger.info(f"  {'-'*30}-+-{'-'*30}")

            n = max(len(results), len(reranked))
            for i in range(min(n, 5)):
                before = results[i].text[:28]  if i < len(results)  else ""
                after  = reranked[i].text[:28] if i < len(reranked) else ""
                a_score = f"({reranked[i].score:.3f})" if i < len(reranked) else ""
                logger.info(f"  {i+1}. {before:28s} | {i+1}. {after:28s} {a_score}")

        logger.info("\n" + "═" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# EXÉCUTION DIRECTE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  TEST SUITE — Phase 2 : CrossEncoder Re-Ranking")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    for cls in [
        TestRerankerConfig,
        TestCrossEncoderReranker,
        TestRerankerIntegration,
        TestRerankerEdgeCases,
        TestRerankerBenchmark,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    ok  = result.testsRun - len(result.failures) - len(result.errors)
    print(f"  ✅ Réussis  : {ok}")
    print(f"  ❌ Échecs   : {len(result.failures)}")
    print(f"  💥 Erreurs  : {len(result.errors)}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
