from src.framework.llm.llm_factory import build_llm_client
from src.framework.loaders.pdf_loader import PDFLoader
from src.agents.academic_performance_agent.agent import AcademicPerformanceAgent


def main():
    llm = build_llm_client()
    agent = AcademicPerformanceAgent(llm=llm)
    loader = PDFLoader()

    print("=" * 50)
    print("Academic Performance Agent - Web Research")
    print("Type 'exit' to quit.")
    print("=" * 50)

    doc_path = input(
        "\nOptional path to YOUR result sheet PDF (leave blank to skip): "
    ).strip()
    user_document = loader.load(doc_path) if doc_path else ""

    while True:
        question = input("\nAsk a question: ")
        if question.lower() == "exit":
            break

        response = agent.process_query(
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
