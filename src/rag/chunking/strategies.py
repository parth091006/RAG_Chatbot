from __future__ import annotations


def fixed_size_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into fixed-size chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()
    chunks: list[str] = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break

    return chunks


def recursive_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Fallback chunking strategy that uses fixed-size chunking by default."""
    return fixed_size_chunks(text, chunk_size=chunk_size, overlap=overlap)
