from pypdf import PdfReader

from src.framework.interfaces.interfaces import IDocumentLoader
from src.framework.utils.logger import Logger


class PDFLoader(IDocumentLoader):
    """Safely extracts text from PDF files."""

    def load(self, file_path: str, max_pages: int = 15, max_chars: int = 30000) -> str:
        Logger.info(f"Loading PDF: {file_path}")
        reader = PdfReader(file_path)
        if len(reader.pages) > max_pages:
            raise ValueError(f"PDF exceeds the {max_pages}-page limit.")

        parts: list[str] = []
        total = 0
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text:
                remaining = max_chars - total
                if remaining <= 0:
                    break
                piece = page_text[:remaining]
                parts.append(piece)
                total += len(piece)

        text = "\n".join(parts).strip()
        if not text:
            raise ValueError("No readable text was found in the PDF.")
        Logger.success("PDF Loaded Successfully.")
        return text
