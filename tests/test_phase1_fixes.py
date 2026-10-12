from __future__ import annotations

from pathlib import Path

from src.rag.chunking.chunker import section_aware_chunks
from src.rag.embeddings.embedder import Embedder
from src.rag.generation.generator import Generator
from src.rag.ingestion.metadata import DocumentMetadata
from src.rag.retrieval.dense import DenseRetriever


def test_section_aware_chunks_handles_dict_pages() -> None:
    pages = [
        {"text": "2.1 The key-query-value self-attention mechanism Q and K determine attention weights."},
        {"text": "3.1 Multi-head Self-Attention Multiple heads capture different relationships."},
    ]
    chunks = section_aware_chunks(
        pages,
        document_id="doc001",
        document_name="paper.pdf",
        chunk_size=50,
        overlap=5,
    )
    assert len(chunks) == 2
    assert chunks[0]["section"] == "2.1 The key-query-value self-attention mechanism"
    assert chunks[1]["section"] == "3.1 Multi-head Self-Attention"


def test_embedder_preserves_short_domain_tokens() -> None:
    embedder = Embedder("bow")
    tokens = embedder._tokenize("What are Q, K, and V in self-attention with AI and ML?")
    assert "q" in tokens
    assert "k" in tokens
    assert "v" in tokens
    assert "ai" in tokens
    assert "ml" in tokens
    assert "self-attention" in tokens


def test_generator_preserves_short_domain_tokens() -> None:
    terms = Generator._extract_terms("Evaluate Q, K, V and F1 score in ML models")
    assert "q" in terms
    assert "k" in terms
    assert "v" in terms
    assert "f1" in terms
    assert "ml" in terms


def test_generator_split_sentences_protects_abbreviations_and_numbers() -> None:
    text = (
        "Researchers use self-attention, e.g., in Transformers (see Fig. 1 and Eq. 3.14). "
        "The model achieves high accuracy as shown by Bahdanau et al. in machine translation."
    )
    sentences = Generator._split_sentences(text)
    assert len(sentences) == 2
    assert "e.g." in sentences[0]
    assert "Fig. 1" in sentences[0]
    assert "Eq. 3.14" in sentences[0]
    assert "Bahdanau et al." in sentences[1]


def test_document_metadata_uses_stem_and_reads_pages() -> None:
    pdf_path = Path("data/raw/self-attention-transformers-2023.pdf")
    meta = DocumentMetadata.from_file(pdf_path)
    assert meta.document_id == "doc_self-attention-transformers-2023"
    assert meta.filename == "self-attention-transformers-2023.pdf"
    assert meta.pages == 18


def test_dense_retriever_zero_query_vector() -> None:
    retriever = DenseRetriever([[1.0, 0.0], [0.0, 1.0]])
    ranked = retriever.search([0.0, 0.0], top_k=2)
    assert len(ranked) == 2
    assert ranked[0][1] == 0.0
