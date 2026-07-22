# tests/test_orchestrator.py
"""
Tests d'intégration pour Phase 4 – LangGraph Orchestrator (Ollama).
"""

import json
import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 0 : Prérequis (Ollama + LangGraph)
# ─────────────────────────────────────────────────────────────────────────────

def test_prerequisites():
    print("\n" + "="*60)
    print("SUITE 0 : Prérequis")
    print("="*60)

    # LangGraph
    try:
        from langgraph.graph import StateGraph
        import langgraph
        print(f"✓ LangGraph : {langgraph.__version__}")
    except ImportError:
        print("❌ LangGraph manquant → pip install langgraph")
        return False

    # langchain-ollama
    try:
        from langchain_ollama import ChatOllama
        print("✓ langchain-ollama disponible")
    except ImportError:
        print("❌ langchain-ollama → pip install langchain-ollama")
        return False

    # Ollama server
    from agents.ollama_client import OllamaClientFactory
    if OllamaClientFactory.ping():
        models = OllamaClientFactory.list_local_models()
        print(f"✓ Ollama server UP | Modèles : {models}")
    else:
        print("⚠  Ollama non accessible. Lancez 'ollama serve'")
        print("   Certains tests seront sautés")

    print("\n✅ Suite 0 OK")
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 1 : OllamaClientFactory
# ─────────────────────────────────────────────────────────────────────────────

def test_ollama_client():
    print("\n" + "="*60)
    print("SUITE 1 : OllamaClientFactory")
    print("="*60)

    from agents.ollama_client import OllamaClientFactory

    if not OllamaClientFactory.ping():
        print("⚠  Ollama non démarré – test sauté")
        return

    # Lister les modèles
    models = OllamaClientFactory.list_local_models()
    print(f"✓ Modèles disponibles : {models}")

    # Sélection automatique
    best = OllamaClientFactory.get_best_available_model()
    print(f"✓ Meilleur modèle sélectionné : {best}")

    # Créer un LLM et faire un appel simple
    llm = OllamaClientFactory.create(temperature=0.0)
    from langchain_core.messages import HumanMessage
    t0 = time.time()
    response = llm.invoke([HumanMessage(content="Réponds juste 'OK' en un mot")])
    elapsed = (time.time() - t0) * 1000
    print(f"✓ Inférence Ollama : '{response.content.strip()}' en {elapsed:.0f}ms")
    assert len(response.content) > 0

    print("\n✅ Suite 1 PASSÉE")


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 2 : AgentState TypedDict
# ─────────────────────────────────────────────────────────────────────────────

def test_agent_state():
    print("\n" + "="*60)
    print("SUITE 2 : AgentState")
    print("="*60)

    from agents.state import AgentState
    from langchain_core.messages import HumanMessage

    state: AgentState = {
        "query": "test query",
        "video_id_filter": None,
        "language": "fr",
        "messages": [HumanMessage(content="test")],
        "search_strategy": "text",
        "router_reasoning": "",
        "search_results": [],
        "visual_results": [],
        "final_answer": "",
        "sources": [],
        "steps_taken": [],
        "error": None,
        "total_time_ms": 0.0,
    }

    assert state["query"] == "test query"
    assert state["search_strategy"] == "text"
    print("✓ AgentState créé et validé")
    print(f"✓ Champs : {list(state.keys())}")

    print("\n✅ Suite 2 PASSÉE")


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 3 : Tools
# ─────────────────────────────────────────────────────────────────────────────

def test_tools():
    print("\n" + "="*60)
    print("SUITE 3 : Tools")
    print("="*60)

    from agents.tools import (
        ALL_TOOLS,
        get_database_stats,
        get_indexed_videos,
        search_video_segments,
    )

    print(f"✓ {len(ALL_TOOLS)} outils disponibles : {[t.name for t in ALL_TOOLS]}")

    # Test get_database_stats (ne nécessite pas de données)
    t0 = time.time()
    result = get_database_stats.invoke({})
    elapsed = (time.time() - t0) * 1000
    data = json.loads(result)
    print(f"✓ get_database_stats : {data} ({elapsed:.0f}ms)")

    # Test get_indexed_videos
    result = get_indexed_videos.invoke({})
    data = json.loads(result)
    print(f"✓ get_indexed_videos : {data.get('total_videos', 0)} vidéos")

    # Test search_video_segments (retourne [] si base vide)
    result = search_video_segments.invoke({
        "query": "intelligence artificielle",
        "top_k": 3,
        "use_reranker": False,
    })
    data = json.loads(result)
    print(f"✓ search_video_segments : {len(data) if isinstance(data, list) else data}")

    print("\n✅ Suite 3 PASSÉE")


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 4 : Graphe LangGraph (sans Ollama)
# ─────────────────────────────────────────────────────────────────────────────

def test_graph_structure():
    """Vérifie que le graphe est correctement compilé (sans lancer Ollama)."""
    print("\n" + "="*60)
    print("SUITE 4 : Structure du graphe LangGraph")
    print("="*60)

    from agents.ollama_client import OllamaClientFactory

    if not OllamaClientFactory.ping():
        print("⚠  Ollama requis pour ce test – sauté")
        return

    from agents.orchestrator import VideoSearchOrchestrator

    orc = VideoSearchOrchestrator(verbose=True)

    # Vérifier le graphe compilé
    graph = orc.graph
    assert graph is not None
    print("✓ Graphe compilé avec succès")

    # Afficher les nœuds
    nodes = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else ["compilé"]
    print(f"✓ Nœuds : {nodes}")

    print("\n✅ Suite 4 PASSÉE")


# ─────────────────────────────────────────────────────────────────────────────
#  Suite 5 : Run complet (nécessite Ollama + données indexées)
# ─────────────────────────────────────────────────────────────────────────────

def test_full_run():
    """Test end-to-end de l'agent avec Ollama."""
    print("\n" + "="*60)
    print("SUITE 5 : Run complet (Ollama requis)")
    print("="*60)

    from agents.ollama_client import OllamaClientFactory

    if not OllamaClientFactory.ping():
        print("⚠  Ollama non démarré – test sauté")
        print("   Lancez : ollama serve && ollama pull llama3.2:3b")
        return

    from agents.orchestrator import VideoSearchOrchestrator

    orc = VideoSearchOrchestrator(verbose=True)

    queries = [
        "Qu'est-ce qui est expliqué dans cette vidéo ?",
        "Montre-moi les slides présentées",       # → visual
        "Résume les points clés de la conférence",
    ]

    for query in queries:
        print(f"\n  ► Query : '{query}'")
        t0 = time.time()
        result = orc.run(query=query)
        elapsed = (time.time() - t0) * 1000

        print(f"    Stratégie  : {result.get('search_strategy')}")
        print(f"    Reasoning  : {result.get('router_reasoning', '')[:80]}")
        print(f"    Steps      : {result.get('steps_taken')}")
        print(f"    Sources    : {len(result.get('sources', []))} références")
        print(f"    Durée      : {elapsed:.0f}ms")
        print(f"    Réponse    : {result.get('final_answer', '')[:200]}…")

        assert result.get("final_answer"), "final_answer vide !"
        assert result.get("steps_taken"), "steps_taken vide !"

    print("\n✅ Suite 5 PASSÉE")


# ─────────────────────────────────────────────────────────────────────────────
#  Runner principal
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "🧠 " * 20)
    print("TESTS – Phase 4 : LangGraph Orchestrator (Ollama)")
    print("🧠 " * 20)

    results = {}
    for name, fn in [
        ("Prérequis",        test_prerequisites),
        ("Ollama Client",    test_ollama_client),
        ("AgentState",       test_agent_state),
        ("Tools",            test_tools),
        ("Graphe Structure", test_graph_structure),
        ("Run Complet",      test_full_run),
    ]:
        try:
            fn()
            results[name] = "✅ PASSÉ"
        except Exception as e:
            results[name] = f"❌ ÉCHOUÉ : {e}"
            logger.exception(f"Erreur dans {name}")

    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    for name, status in results.items():
        print(f"  {status:<50} {name}")

    n = sum(1 for v in results.values() if v.startswith("✅"))
    print(f"\n{n}/{len(results)} tests réussis")
