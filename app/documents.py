import io
import re

from pypdf import PdfReader

from app.schemas import ChunkStrategy


def extract_text(data: bytes, extension: str) -> str:
    if extension == ".txt":
        return data.decode("utf-8-sig")
    reader = PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported")
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def chunk_text(text: str, strategy: ChunkStrategy, size: int, overlap: int) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError("Overlap must be smaller than chunk size")
    text = text.strip()
    if strategy == ChunkStrategy.fixed:
        return [
            text[i : i + size]
            for i in range(0, len(text), size - overlap)
            if text[i : i + size].strip()
        ]
    # Prefer paragraph boundaries; split oversized paragraphs with bounded windows.
    chunks: list[str] = []
    current = ""
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if current and len(current) + len(paragraph) + 2 <= size:
            current += "\n\n" + paragraph
            continue
        if current:
            chunks.append(current)
            prefix = current[-overlap:] if overlap else ""
            current = prefix + "\n\n" + paragraph if prefix else paragraph
        else:
            current = paragraph
        while len(current) > size:
            chunks.append(current[:size])
            current = current[size - overlap :]
    if current.strip():
        chunks.append(current)
    return chunks
