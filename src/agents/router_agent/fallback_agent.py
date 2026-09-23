from src.framework.core.data_models import RAGResponse
from src.framework.interfaces.interfaces import IAgent, ILLMClient


GENERAL_FALLBACK_MESSAGE = (
    "I can help with academic performance, GPA/results, CVs, careers, "
    "jobs, skills, and current industry information."
)


class FallbackAgent(IAgent):
    """Handles unrelated questions while still using live web research."""

    def __init__(self, llm: ILLMClient):
        self.llm = llm

    def process_query(self, query: str, user_document: str = "") -> RAGResponse:
        answer = self.llm.generate(
            prompt=query,
            system_instruction=f"""
You are the general assistant for CareerGuide AI.

{GENERAL_FALLBACK_MESSAGE}

Use live web-search grounding for current/public factual information.
Prefer authoritative sources. If the question is outside the application's
career/academic scope, politely say that it is outside the main scope.
Do not use ChromaDB or any stored PDF knowledge base.
""",
        )

        return RAGResponse(
            question=query,
            retrieved_chunks=[],
            answer=answer,
            sources=getattr(self.llm, "last_sources", []) or [],
        )
