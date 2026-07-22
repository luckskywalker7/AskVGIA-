# tests/test_hybrid_search.py
"""
Tests d'intégration — Phase 1 : Hybrid Search
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Suites de tests :
  1. TestSparseEmbeddingEngine  – BM25 / TF-IDF
  2. TestRRFFusion              – algorithme de fusion
  3. TestVideoStoreHybrid       – indexation + recherche Qdrant
  4. TestHybridSearchEngine     – intégration bout en bout
  5. TestSearchBenchmark        – comparaison des 3 stratégies

Usage :
    python -m pytest tests/test_hybrid_search.py -v
    python -m pytest tests/test_hybrid_search.py -v -k "TestRRF"
    python tests/test_hybrid_search.py            # mode script direct
"""

from __future__ import annotations

import logging
import random
import sys
import time
import unittest
import shutil
import tempfile
from pathlib import Path

# Patch pour NumPy 2.0 (suppression de np.NINF)
import numpy as np

if not hasattr(np, 'NINF'):
    np.NINF = -np.inf

# ── Ajout du répertoire racine au PYTHONPATH ──────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.sparse_embeddings import SparseEmbedding, SparseEmbeddingEngine
from core.storage import SearchResult, VideoStore
from core.search_strategies import (
    HybridSearchEngine,
    RRFFusion,
    SearchConfig,
    SearchMode,
    TemporalBooster,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_hybrid")


# ──────────────────────────────────────────────────────────────────────────────
# CLASSE DE BASE AVEC DOSSIER QDRANT UNIQUE
# ──────────────────────────────────────────────────────────────────────────────

class BaseTestCase(unittest.TestCase):
    """Classe de base avec dossier Qdrant unique par classe de test."""

    @classmethod
    def setUpClass(cls):
        """Crée un dossier temporaire unique pour cette classe de test."""
        # Créer un dossier temporaire unique
        cls.temp_dir = tempfile.mkdtemp(prefix=f"qdrant_{cls.__name__}_")
        cls.qdrant_path = Path(cls.temp_dir)

        logger.info(f"📁 Dossier Qdrant temporaire créé: {cls.qdrant_path}")

        # Initialiser VideoStore avec ce chemin
        from core.storage import VideoStore
        cls.store = VideoStore(path=str(cls.qdrant_path))

    @classmethod
    def tearDownClass(cls):
        """Nettoie après les tests."""
        if hasattr(cls, 'store') and cls.store:
            try:
                # Fermer proprement la connexion
                if hasattr(cls.store, 'client'):
                    cls.store.client.close()
                cls.store = None
                logger.info(f"🔒 Connexion Qdrant fermée")
            except Exception as e:
                logger.warning(f"⚠️ Erreur lors de la fermeture: {e}")

        # Attendre que les verrous soient libérés
        time.sleep(0.5)

        # Supprimer le dossier temporaire
        if hasattr(cls, 'temp_dir') and cls.temp_dir:
            try:
                shutil.rmtree(cls.temp_dir)
                logger.info(f"🧹 Dossier temporaire {cls.temp_dir} supprimé")
            except Exception as e:
                logger.warning(f"⚠️ Impossible de supprimer {cls.temp_dir}: {e}")
                # Forcer la suppression sur Windows
                import os
                os.system(f'rmdir /s /q "{cls.temp_dir}" 2>nul')


# ──────────────────────────────────────────────────────────────────────────────
# FIXTURES PARTAGÉES
# ──────────────────────────────────────────────────────────────────────────────

DIM = 768  # dimension dense

# Corpus de test (EN + AR pour tester le multilingue)
CORPUS = [
    {
        "text": "Introduction to artificial intelligence and machine learning",
        "start_time": 0.0, "end_time": 30.0,
        "language": "en", "speaker": "SPEAKER_00",
        "topic": "AI overview",
    },
    {
        "text": "Deep learning uses neural networks with multiple hidden layers",
        "start_time": 30.0, "end_time": 60.0,
        "language": "en", "speaker": "SPEAKER_01",
        "topic": "Deep learning",
    },
    {
        "text": "Transformer architecture revolutionized natural language processing",
        "start_time": 60.0, "end_time": 90.0,
        "language": "en", "speaker": "SPEAKER_00",
        "topic": "NLP",
    },
    {
        "text": "Generative AI models can create realistic text, images and audio",
        "start_time": 90.0, "end_time": 120.0,
        "language": "en", "speaker": "SPEAKER_01",
        "topic": "Generative AI",
    },
    {
        "text": "Reinforcement learning trains agents through reward and punishment",
        "start_time": 120.0, "end_time": 150.0,
        "language": "en", "speaker": "SPEAKER_00",
        "topic": "RL",
    },
    {
        "text": "Computer vision enables machines to interpret and understand images",
        "start_time": 150.0, "end_time": 180.0,
        "language": "en", "speaker": "SPEAKER_01",
        "topic": "Computer vision",
    },
    {
        "text": "الذكاء الاصطناعي يغير مستقبل التكنولوجيا والصناعة",
        "start_time": 180.0, "end_time": 210.0,
        "language": "ar", "speaker": "SPEAKER_00",
        "topic": "AI Arabic",
    },
    {
        "text": "Large language models like GPT use billions of parameters",
        "start_time": 210.0, "end_time": 240.0,
        "language": "en", "speaker": "SPEAKER_01",
        "topic": "LLM",
    },
]

VIDEO_ID = "test_video_hybrid_001"


def make_dense_vecs(n: int, dim: int = DIM, seed: int = 42) -> list:
    rng = random.Random(seed)
    return [[rng.uniform(-1, 1) for _ in range(dim)] for _ in range(n)]


def make_dense_query(dim: int = DIM, seed: int = 99) -> list:
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(dim)]


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 1 – SparseEmbeddingEngine
# ──────────────────────────────────────────────────────────────────────────────

class TestSparseEmbeddingEngine(unittest.TestCase):
    """Tests du moteur d'embeddings sparse (BM25 + fallback TF-IDF)."""

    @classmethod
    def setUpClass(cls):
        cls.engine = SparseEmbeddingEngine()
        logger.info(f"🔧 SparseEmbeddingEngine : backend={cls.engine.backend!r}")

    def test_engine_initialized(self):
        """L'engine doit s'initialiser sans erreur."""
        self.assertTrue(self.engine.is_ready)
        self.assertIn(self.engine.backend, ("fastembed", "tfidf"))

    def test_encode_single(self):
        """Encoder un texte retourne un SparseEmbedding valide."""
        emb = self.engine.encode("deep learning neural networks")
        self.assertIsInstance(emb, SparseEmbedding)
        self.assertEqual(len(emb.indices), len(emb.values))

    def test_encode_empty_string(self):
        """Une chaîne vide ne doit pas lever d'exception."""
        emb = self.engine.encode("")
        self.assertIsInstance(emb, SparseEmbedding)

    def test_encode_batch(self):
        """Le batch retourne autant d'embeddings que de textes."""
        texts = [s["text"] for s in CORPUS]
        embs = self.engine.encode_batch(texts)
        self.assertEqual(len(embs), len(texts))
        for emb in embs:
            self.assertIsInstance(emb, SparseEmbedding)
            self.assertEqual(len(emb.indices), len(emb.values))

    def test_to_qdrant_conversion(self):
        """to_qdrant() retourne un objet SparseVector Qdrant."""
        emb = self.engine.encode("transformer attention mechanism")
        if not emb.is_empty():
            sv = emb.to_qdrant()
            self.assertTrue(hasattr(sv, "indices"))
            self.assertTrue(hasattr(sv, "values"))

    def test_different_texts_different_vectors(self):
        """Deux textes différents doivent produire des vecteurs différents."""
        e1 = self.engine.encode("machine learning models")
        e2 = self.engine.encode("cooking pasta recipe")
        # Les indices non-nuls doivent différer
        set1 = set(e1.indices)
        set2 = set(e2.indices)
        # Il y a forcément des termes différents
        self.assertNotEqual(set1, set2)

    def test_multilingual_arabic(self):
        """Encoder un texte arabe ne doit pas lever d'exception."""
        emb = self.engine.encode("الذكاء الاصطناعي يغير التكنولوجيا")
        self.assertIsInstance(emb, SparseEmbedding)


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 2 – RRFFusion
# ──────────────────────────────────────────────────────────────────────────────

class TestRRFFusion(unittest.TestCase):
    """Tests de l'algorithme Reciprocal Rank Fusion."""

    def _make_result(self, sid: str, score: float) -> SearchResult:
        return SearchResult(
            segment_id=sid, video_id="v1", text=f"text_{sid}",
            start_time=0.0, end_time=10.0, timestamp_formatted="00:00",
            score=score, language="en", speaker="", topic="", chunk_type="",
        )

    def test_empty_lists(self):
        """Fusion de listes vides → résultat vide."""
        fuser = RRFFusion()
        self.assertEqual(fuser.fuse([], top_k=5), [])
        self.assertEqual(fuser.fuse([[]], top_k=5), [])

    def test_single_list(self):
        """Fusion d'une seule liste retourne les bons résultats."""
        fuser = RRFFusion(k=60)
        results = [self._make_result(f"seg_{i}", 1.0 - i * 0.1) for i in range(5)]
        fused = fuser.fuse([results], top_k=3)
        self.assertEqual(len(fused), 3)

    def test_scores_normalized(self):
        """Les scores normalisés doivent être dans [0, 1]."""
        fuser = RRFFusion(k=60)
        r1 = [self._make_result(f"A{i}", 0.9 - i * 0.1) for i in range(5)]
        r2 = [self._make_result(f"B{i}", 0.8 - i * 0.1) for i in range(5)]
        fused = fuser.fuse([r1, r2], top_k=10)
        for r in fused:
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 1.0 + 1e-9)

    def test_top_result_appears_in_both_lists(self):
        """
        Un segment présent dans les 2 listes (rang 1) doit avoir
        le score RRF le plus élevé.
        """
        fuser = RRFFusion(k=60)
        shared = self._make_result("shared_top", 0.95)
        only_d1 = self._make_result("only_dense", 0.90)
        only_s1 = self._make_result("only_sparse", 0.85)

        dense_list = [shared, only_d1]
        sparse_list = [shared, only_s1]

        fused = fuser.fuse([dense_list, sparse_list], top_k=5)
        self.assertEqual(fused[0].segment_id, "shared_top")

    def test_rrf_k_effect(self):
        """k plus petit → plus de discrimination entre rangs proches."""
        r1 = [self._make_result(f"seg_{i}", 1.0 - i * 0.1) for i in range(10)]
        r2 = [self._make_result(f"seg_{i}", 1.0 - i * 0.1) for i in range(10)]

        fuser_strict = RRFFusion(k=10)
        fuser_smooth = RRFFusion(k=100)

        f1 = fuser_strict.fuse([r1, r2], top_k=5)
        f2 = fuser_smooth.fuse([r1, r2], top_k=5)

        # Les deux doivent retourner les mêmes segments (ordre identique ici)
        ids1 = [r.segment_id for r in f1]
        ids2 = [r.segment_id for r in f2]
        self.assertEqual(ids1, ids2)

    def test_weights_influence_ranking(self):
        """
        Donner plus de poids à la liste sparse doit faire monter
        un segment présent uniquement dans la liste sparse.
        """
        fuser = RRFFusion(k=1)  # k=1 pour amplifier les différences

        top_dense = self._make_result("dense_only", 0.99)
        top_sparse = self._make_result("sparse_top", 0.99)

        dense_list = [top_dense]
        sparse_list = [top_sparse]

        # Poids equal → ordre indéfini entre les deux (tous rang 1)
        # Poids sparse fort → sparse_top doit gagner
        fused = fuser.fuse(
            [dense_list, sparse_list],
            weights=[1.0, 10.0],
            top_k=2,
        )
        self.assertEqual(fused[0].segment_id, "sparse_top")

    def test_top_k_limit(self):
        """Le résultat ne doit jamais dépasser top_k éléments."""
        fuser = RRFFusion()
        results = [self._make_result(f"seg_{i}", 1.0) for i in range(20)]
        fused = fuser.fuse([results], top_k=7)
        self.assertLessEqual(len(fused), 7)


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 3 – VideoStore Hybrid
# ──────────────────────────────────────────────────────────────────────────────

class TestVideoStoreHybrid(BaseTestCase):
    """Tests d'intégration VideoStore avec schéma hybrid."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.engine = SparseEmbeddingEngine()

        # Reset propre pour les tests
        cls.store.reset()
        logger.info("🔧 VideoStore réinitialisé (schéma hybrid)")

        # Préparer les vecteurs
        cls.dense_vecs = make_dense_vecs(len(CORPUS))
        texts = [s["text"] for s in CORPUS]
        cls.sparse_vecs = cls.engine.encode_batch(texts)

        # Indexer
        cls.n_indexed = cls.store.add_video(
            video_id=VIDEO_ID,
            segments=CORPUS,
            dense_embeddings=cls.dense_vecs,
            sparse_embeddings=cls.sparse_vecs,
        )

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_video(VIDEO_ID)
        logger.info("🧹 Données de test supprimées")
        super().tearDownClass()

    def test_indexation_count(self):
        """Le nombre de points indexés doit correspondre au corpus."""
        self.assertEqual(self.n_indexed, len(CORPUS))

    def test_store_count_global(self):
        """store.count() doit retourner le nombre total de points."""
        count = self.store.count()
        self.assertGreaterEqual(count, len(CORPUS))

    def test_store_count_by_video(self):
        """store.count(video_id) doit retourner les points de cette vidéo."""
        count = self.store.count(VIDEO_ID)
        self.assertEqual(count, len(CORPUS))

    def test_list_videos_contains_test(self):
        """list_videos() doit contenir le VIDEO_ID de test."""
        videos = self.store.list_videos()
        self.assertIn(VIDEO_ID, videos)

    def test_search_dense_returns_results(self):
        """search_dense() doit retourner des SearchResult."""
        query_vec = make_dense_query()
        results = self.store.search_dense(
            query_vector=query_vec,
            top_k=5,
            payload_filter=VideoStore.build_filter(video_id=VIDEO_ID),
        )
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)
        for r in results:
            self.assertIsInstance(r, SearchResult)
            self.assertEqual(r.video_id, VIDEO_ID)

    def test_search_sparse_returns_results(self):
        """search_sparse() doit retourner des SearchResult."""
        sparse_query = self.engine.encode("deep learning neural networks")
        if sparse_query.is_empty():
            self.skipTest("Sparse query vide (backend TF-IDF non fitté)")

        results = self.store.search_sparse(
            query_sparse=sparse_query,
            top_k=5,
            payload_filter=VideoStore.build_filter(video_id=VIDEO_ID),
        )
        self.assertIsInstance(results, list)

    def test_schema_is_hybrid(self):
        """Le schéma de la collection doit être 'hybrid'."""
        self.assertTrue(self.store._is_hybrid_schema())

    def test_stats(self):
        """get_stats() doit retourner un StoreStats cohérent."""
        stats = self.store.get_stats()
        self.assertEqual(stats.schema, "hybrid")
        self.assertGreaterEqual(stats.total_points, len(CORPUS))
        self.assertGreaterEqual(stats.total_videos, 1)

    def test_build_filter_video_id(self):
        """build_filter avec video_id doit retourner un Filter Qdrant."""
        from qdrant_client.models import Filter
        f = VideoStore.build_filter(video_id=VIDEO_ID)
        self.assertIsNotNone(f)
        self.assertIsInstance(f, Filter)

    def test_build_filter_none_when_no_args(self):
        """build_filter sans args doit retourner None."""
        f = VideoStore.build_filter()
        self.assertIsNone(f)

    def test_dense_only_indexation(self):
        """
        Indexer sans sparse_embeddings (mode legacy) ne doit pas lever d'erreur.
        """
        tmp_id = "test_dense_only_tmp"
        try:
            n = self.store.add_video(
                video_id=tmp_id,
                segments=CORPUS[:2],
                dense_embeddings=self.dense_vecs[:2],
                sparse_embeddings=None,  # ← dense-only
            )
            self.assertEqual(n, 2)
        finally:
            self.store.delete_video(tmp_id)

    def test_delete_video(self):
        """Supprimer une vidéo doit réduire le count."""
        tmp_id = "test_delete_tmp"
        self.store.add_video(
            tmp_id,
            CORPUS[:3],
            self.dense_vecs[:3],
            self.sparse_vecs[:3],
        )
        before = self.store.count(tmp_id)
        self.assertEqual(before, 3)

        ok = self.store.delete_video(tmp_id)
        self.assertTrue(ok)
        self.assertEqual(self.store.count(tmp_id), 0)


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 4 – HybridSearchEngine (bout en bout)
# ──────────────────────────────────────────────────────────────────────────────

class TestHybridSearchEngine(BaseTestCase):
    """Tests d'intégration bout-en-bout du moteur hybride."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.sparse = SparseEmbeddingEngine()
        cls.engine = HybridSearchEngine(cls.store, cls.sparse)

        # Nettoyer + indexer
        cls.store.reset()
        dense_vecs = make_dense_vecs(len(CORPUS))
        sparse_vecs = cls.sparse.encode_batch([s["text"] for s in CORPUS])
        cls.store.add_video(VIDEO_ID, CORPUS, dense_vecs, sparse_vecs)

        cls.query_dense = make_dense_query()
        cls.query_text = "deep learning neural networks transformer"

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_video(VIDEO_ID)
        super().tearDownClass()

    # ── Modes individuels ──────────────────────────────────────────────────

    def test_dense_mode(self):
        cfg = SearchConfig(mode=SearchMode.DENSE, top_k=5, video_id=VIDEO_ID)
        results = self.engine.search(self.query_text, self.query_dense, cfg)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)
        for r in results:
            self.assertEqual(r.strategy, "dense")

    def test_sparse_mode(self):
        cfg = SearchConfig(mode=SearchMode.SPARSE, top_k=5, video_id=VIDEO_ID)
        results = self.engine.search(self.query_text, self.query_dense, cfg)
        self.assertIsInstance(results, list)
        # Pas de crash même si BM25 retourne 0 résultats

    def test_hybrid_mode(self):
        cfg = SearchConfig(mode=SearchMode.HYBRID, top_k=5, video_id=VIDEO_ID)
        results = self.engine.search(self.query_text, self.query_dense, cfg)
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), 5)
        for r in results:
            self.assertEqual(r.strategy, "hybrid")

    def test_hybrid_scores_in_range(self):
        """Scores hybrides normalisés doivent être dans [0, 1]."""
        cfg = SearchConfig(mode=SearchMode.HYBRID, top_k=5, video_id=VIDEO_ID)
        results = self.engine.search(self.query_text, self.query_dense, cfg)
        for r in results:
            self.assertGreaterEqual(r.score, 0.0)
            self.assertLessEqual(r.score, 1.0 + 1e-9)

    def test_top_k_respected(self):
        """Le nombre de résultats ne doit pas dépasser top_k."""
        for mode in SearchMode:
            cfg = SearchConfig(mode=mode, top_k=3, video_id=VIDEO_ID)
            results = self.engine.search(self.query_text, self.query_dense, cfg)
            self.assertLessEqual(
                len(results), 3,
                msg=f"Mode {mode.value} retourne {len(results)} > top_k=3"
            )

    def test_results_have_required_fields(self):
        """Chaque résultat doit avoir tous les champs obligatoires."""
        cfg = SearchConfig(mode=SearchMode.HYBRID, top_k=5, video_id=VIDEO_ID)
        results = self.engine.search(self.query_text, self.query_dense, cfg)
        for r in results:
            self.assertTrue(r.segment_id, "segment_id vide")
            self.assertTrue(r.video_id, "video_id vide")
            self.assertIsNotNone(r.text)
            self.assertIsInstance(r.start_time, float)
            self.assertIsInstance(r.end_time, float)
            self.assertIsInstance(r.score, float)

    def test_filter_by_language(self):
        """Filtrer par language='ar' doit retourner uniquement des résultats AR."""
        cfg = SearchConfig(
            mode=SearchMode.DENSE, top_k=5,
            video_id=VIDEO_ID, language="ar",
        )
        results = self.engine.search(self.query_text, self.query_dense, cfg)
        for r in results:
            self.assertEqual(r.language, "ar",
                             f"Attendu 'ar', obtenu '{r.language}'")

    def test_temporal_boost_no_crash(self):
        """Le boosting temporel ne doit pas lever d'exception."""
        cfg = SearchConfig(
            mode=SearchMode.HYBRID, top_k=5,
            video_id=VIDEO_ID,
            temporal_boost=True,
            temporal_window=60.0,
        )
        results = self.engine.search(self.query_text, self.query_dense, cfg)
        self.assertIsInstance(results, list)

    def test_default_config_is_hybrid(self):
        """Sans config explicite, le mode par défaut doit être HYBRID."""
        cfg = SearchConfig()
        self.assertEqual(cfg.mode, SearchMode.HYBRID)


# ──────────────────────────────────────────────────────────────────────────────
# SUITE 5 – Benchmark comparatif
# ──────────────────────────────────────────────────────────────────────────────

class TestSearchBenchmark(BaseTestCase):
    """
    Benchmark des 3 stratégies sur des requêtes représentatives.
    Pas un test de précision absolue (vecteurs aléatoires),
    mais vérifie la robustesse et mesure les temps d'exécution.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.sparse = SparseEmbeddingEngine()
        cls.engine = HybridSearchEngine(cls.store, cls.sparse)

        cls.store.reset()
        dense_vecs = make_dense_vecs(len(CORPUS))
        sparse_vecs = cls.sparse.encode_batch([s["text"] for s in CORPUS])
        cls.store.add_video(VIDEO_ID, CORPUS, dense_vecs, sparse_vecs)

    @classmethod
    def tearDownClass(cls):
        cls.store.delete_video(VIDEO_ID)
        super().tearDownClass()

    TEST_QUERIES = [
        ("deep learning layers", make_dense_query(seed=10)),
        ("generative AI text images", make_dense_query(seed=20)),
        ("transformer attention NLP", make_dense_query(seed=30)),
        ("الذكاء الاصطناعي", make_dense_query(seed=40)),
        ("reinforcement reward agent", make_dense_query(seed=50)),
    ]

    def test_benchmark_all_queries(self):
        """
        Exécute benchmark() sur chaque requête test,
        vérifie que les 3 modes retournent des résultats cohérents.
        """
        logger.info("\n" + "═" * 65)
        logger.info("  BENCHMARK — Comparaison Dense / Sparse / Hybrid")
        logger.info("═" * 65)

        for query_text, query_dense in self.TEST_QUERIES:
            with self.subTest(query=query_text):
                comparison = self.engine.benchmark(
                    query=query_text,
                    query_dense=query_dense,
                    top_k=3,
                    video_id=VIDEO_ID,
                )
                self.assertIn("dense", comparison)
                self.assertIn("sparse", comparison)
                self.assertIn("hybrid", comparison)

    def test_search_time_acceptable(self):
        """
        La recherche hybride doit prendre moins de 500 ms
        sur une petite collection (test de performance minimale).
        """
        query_dense = make_dense_query(seed=77)
        query_text = "machine learning computer vision"
        cfg = SearchConfig(
            mode=SearchMode.HYBRID, top_k=5, video_id=VIDEO_ID
        )

        t0 = time.perf_counter()
        results = self.engine.search(query_text, query_dense, cfg)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(f"⏱  Hybrid search : {elapsed:.1f} ms ({len(results)} résultats)")
        self.assertLess(elapsed, 500.0,
                        f"Recherche hybride trop lente : {elapsed:.1f} ms > 500 ms")

    def test_print_comparison_table(self):
        """Affiche un tableau comparatif lisible des 3 stratégies."""
        query_text = "deep learning neural networks"
        query_dense = make_dense_query(seed=42)

        comparison = self.engine.benchmark(
            query=query_text, query_dense=query_dense,
            top_k=3, video_id=VIDEO_ID,
        )

        logger.info("\n\n📊 TABLEAU COMPARATIF")
        logger.info(f"   Requête : '{query_text}'")
        logger.info("─" * 65)

        for mode, results in comparison.items():
            logger.info(f"\n  [{mode.upper():6s}]  {len(results)} résultats")
            for i, r in enumerate(results, 1):
                logger.info(
                    f"    {i}. [{r.timestamp_formatted}] "
                    f"score={r.score:.4f}  {r.text[:50]}"
                )

        logger.info("\n" + "─" * 65)


# ──────────────────────────────────────────────────────────────────────────────
# EXÉCUTION DIRECTE
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  TEST SUITE — Phase 1 : Hybrid Search (Dense + BM25 + RRF)")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Ordre d'exécution logique
    for cls in [
        TestSparseEmbeddingEngine,
        TestRRFFusion,
        TestVideoStoreHybrid,
        TestHybridSearchEngine,
        TestSearchBenchmark,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"  Tests réussis  : {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Échecs         : {len(result.failures)}")
    print(f"  Erreurs        : {len(result.errors)}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
