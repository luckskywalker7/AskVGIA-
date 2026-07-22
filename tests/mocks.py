# tests/mocks.py
"""LLM fictif pour les tests sans Ollama ni OpenAI."""

from __future__ import annotations

import json
from typing import Any, List

from langchain_core.messages import AIMessage


class MockChatLLM:
    """
    LLM de substitution pour les tests.
    Simule un appel d'outil suivi d'une réponse finale.
    """

    def __init__(self, tools: list, max_calls: int = 1):
        self.tools     = {t.name: t for t in tools}
        self.max_calls = max_calls
        self._calls    = 0

    def bind_tools(self, tools) -> "MockChatLLM":
        """Compatibilité avec l'interface LangChain."""
        return self

    def invoke(self, messages: List[Any]) -> AIMessage:
        """
        Alterne entre appel d'outil et réponse finale.
        Premier appel → tool_call(search_video_segments)
        Appels suivants → réponse textuelle finale
        """
        self._calls += 1

        if self._calls <= self.max_calls:
            # Simuler un appel à search_video_segments
            return AIMessage(
                content="",
                tool_calls=[{
                    "id":   f"mock_call_{self._calls}",
                    "name": "search_video_segments",
                    "args": {
                        "query":    _extract_query(messages),
                        "video_id": None,
                        "top_k":    5,
                        "mode":     "hybrid",
                    },
                }],
            )
        else:
            # Réponse finale simulée
            self._calls = 0
            return AIMessage(
                content=(
                    "Based on the search results, the video discusses "
                    "artificial intelligence and machine learning concepts. "
                    "Key topics include deep learning [00:30], transformer "
                    "architecture [01:00], and generative AI [01:30]."
                )
            )


def _extract_query(messages) -> str:
    """Extrait le texte de la dernière HumanMessage."""
    from langchain_core.messages import HumanMessage
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content[:100]
    return "default query"
