# Advanced RAG Research Assistant

This project is a research-oriented Retrieval-Augmented Generation (RAG) application for analyzing academic papers and answering grounded questions with citations.

## Features

- PDF ingestion and parsing
- Metadata-aware chunking
- Embedding generation
- Dense, sparse, and hybrid retrieval
- Reranking
- Citation tracking
- Multi-document comparison
- Evaluation metrics for retrieval and generation
- FastAPI backend
- Dockerized deployment

## Project structure

See the repository layout for the main modules:

- `src/rag/ingestion` for PDF parsing and metadata extraction
- `src/rag/chunking` for chunking strategies
- `src/rag/retrieval` for dense, BM25, hybrid, and reranking
- `src/rag/generation` for prompts and answer synthesis
- `src/rag/evaluation` for retrieval and generation metrics
- `src/rag/api` for the REST API
- `tests` for unit and integration tests

## Getting started

1. Create a virtual environment
2. Install dependencies from `pyproject.toml`
3. Set environment variables from `.env.example`
4. Start the API with `uvicorn src.rag.api.main:app --reload`
5. Upload PDFs into the data pipeline

## Roadmap

- Version 0: environment setup
- Version 1: PDF ingestion
- Version 2: chunking
- Version 3: embeddings
- Version 4: retrieval
- Version 5: full RAG generation
- Version 6: citations
- Version 7: multi-document comparison
- Version 8: evaluation and tracing
- Version 9: API and Docker deployment