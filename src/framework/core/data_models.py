from dataclasses import dataclass, field
from typing import List


@dataclass
class DocumentChunk:
    """Legacy document chunk model kept for compatibility."""
    id: str
    text: str
    source: str


@dataclass
class SearchResult:
    """Legacy RAG result model kept for compatibility."""
    chunks: List[DocumentChunk]


@dataclass
class RAGResponse:
    """Final response returned by an agent."""
    question: str
    retrieved_chunks: List[DocumentChunk] = field(default_factory=list)
    answer: str = ""
    sources: List[dict] = field(default_factory=list)
