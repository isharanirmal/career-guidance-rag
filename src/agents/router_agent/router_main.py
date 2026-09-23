import json

from src.framework.core.data_models import RAGResponse
from src.framework.interfaces.interfaces import IAgent
from src.framework.llm.llm_factory import build_llm_client

from src.agents.academic_performance_agent.agent import AcademicPerformanceAgent
from src.agents.career_path_agent.agent import CareerPathAgent
from src.agents.router_agent.fallback_agent import FallbackAgent
from src.agents.router_agent.routing_rules import (
    ROUTER_SYSTEM_PROMPT,
    VALID_LABELS,
    KEYWORD_RULES,
)


class RouterAgent:
    """Routes questions to specialist agents. No vector database is used."""

    def __init__(self, llm):
        self.llm_client = llm

        self.academic_agent = AcademicPerformanceAgent(llm=self.llm_client)
        self.career_agent = CareerPathAgent(llm=self.llm_client)
        self.fallback_agent = FallbackAgent(llm=self.llm_client)

        self.last_routed_label: str | None = None

        self._agent_registry: dict[str, IAgent] = {
            "academic_performance_agent": self.academic_agent,
            "career_path_agent": self.career_agent,
            "general_fallback": self.fallback_agent,
        }

    def _fast_keyword_route(self, query_clean: str):
        scores = [
            (rule.match_count(query_clean), rule.label)
            for rule in KEYWORD_RULES
        ]

        best_score, best_label = max(scores, key=lambda item: item[0])

        if best_score == 0:
            return None

        top_labels = {label for score, label in scores if score == best_score}
        if len(top_labels) > 1:
            return None

        return best_label

    def _llm_route(self, user_query: str) -> str:
        try:
            raw_response = self.llm_client.generate(
                prompt=user_query,
                system_instruction=ROUTER_SYSTEM_PROMPT,
            )

            cleaned = raw_response.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

            decision = json.loads(cleaned)
            label = decision.get("next_agent", "general_fallback")
            return label if label in VALID_LABELS else "general_fallback"

        except Exception:
            return "general_fallback"

    def route_and_execute(self, user_query: str, user_document: str = "") -> RAGResponse:
        query_clean = user_query.lower()
        selected_label = (
            self._fast_keyword_route(query_clean)
            or self._llm_route(user_query)
        )

        self.last_routed_label = selected_label
        print(f"[ROUTER AGENT LOG] Routing query to: --> {selected_label}")

        selected_agent = self._agent_registry.get(
            selected_label, self.fallback_agent
        )
        return selected_agent.process_query(
            user_query,
            user_document=user_document,
        )


def main():
    from src.framework.loaders.pdf_loader import PDFLoader

    llm = build_llm_client()
    router = RouterAgent(llm=llm)
    loader = PDFLoader()

    print("=" * 50)
    print("Career & Academic Guidance Assistant")
    print("Answers use live web research.")
    print("Type 'exit' to quit.")
    print("=" * 50)

    doc_path = input(
        "\nOptional path to YOUR result sheet / CV PDF (leave blank to skip): "
    ).strip()
    user_document = loader.load(doc_path) if doc_path else ""

    while True:
        question = input("\nAsk a question: ")
        if question.lower() == "exit":
            break

        response = router.route_and_execute(
            question,
            user_document=user_document,
        )
        print("\nAnswer:")
        print(response.answer)

        if response.sources:
            print("\nWeb sources:")
            for source in response.sources:
                print(f"- {source.get('title', 'Source')}: {source.get('url', '')}")


if __name__ == "__main__":
    main()
