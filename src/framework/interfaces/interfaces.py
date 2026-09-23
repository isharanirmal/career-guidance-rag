from abc import ABC, abstractmethod
from typing import List

from src.framework.core.data_models import DocumentChunk, SearchResult, RAGResponse


class ILLMClient(ABC):
    """Interface for the web-grounded LLM."""

    @abstractmethod
    def generate(self, prompt: str, system_instruction: str = "") -> str:
        pass


class IVectorStore(ABC):
    """Legacy interface retained only for compatibility."""

    @abstractmethod
    def add_documents(
        self,
        chunks: List[DocumentChunk],
        collection_name: str,
    ) -> None:
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        collection_name: str,
        k: int = 3,
    ) -> SearchResult:
        pass


class IDocumentLoader(ABC):
    """Interface for reading an optional user-uploaded document."""

    @abstractmethod
    def load(self, file_path: str) -> str:
        pass


class IAgent(ABC):
    """Interface for specialist agents."""

    @abstractmethod
    def process_query(
        self,
        query: str,
        user_document: str = "",
    ) -> RAGResponse:
        pass
