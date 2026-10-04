# RAG Project Phase Commands

This file records the commands to run for each phase of the project and a short explanation of what each command verifies.

---

## Phase 0: Environment Setup

### 1. Create virtual environment
```bash
python -m venv .venv
```
What it does: Creates the local Python environment for the project so dependencies stay isolated.

### 2. Activate virtual environment
```powershell
.\.venv\Scripts\Activate.ps1
```
What it does: Activates the project-specific Python environment so commands use the correct dependencies.

### 3. Install project dependencies
```bash
python -m pip install -U pip
python -m pip install fastapi "httpx<0.28" python-dotenv pypdf pytest
```
What it does: Installs the runtime and test packages required by the RAG app and its validation checks.

### 4. Run the full project tests
```bash
python -m pytest -q
```
What it does: Verifies the project is still healthy after each phase and catches regressions.

---

## Phase 1: PDF to Answer (Checkpoint Flow)

### 1. Check PDF text extraction
```bash
python -c "from pathlib import Path; from src.rag.ingestion.parser import extract_text_from_pdf; text = extract_text_from_pdf(Path('data/raw/self-attention-transformers-2023.pdf')); print(len(text)); print(text[:500])"
```
What it does: Confirms the PDF file is readable and that text is successfully extracted from it.

### 2. Check PDF to chunk conversion
```bash
python -c "from pathlib import Path; from src.rag.pipeline import PhaseOneRAG; rag = PhaseOneRAG(Path('data/raw/self-attention-transformers-2023.pdf')); chunks = rag.pdf_to_chunks(); print(len(chunks)); print(chunks[0][:300])"
```
What it does: Verifies the extracted text is split into manageable chunks for retrieval.

### 3. Check chunk to embedding conversion
```bash
python -c "from pathlib import Path; from src.rag.pipeline import PhaseOneRAG; rag = PhaseOneRAG(Path('data/raw/self-attention-transformers-2023.pdf')); rag.pdf_to_chunks(); embeds = rag.chunks_to_embeddings(rag.chunks); print(len(embeds)); print(len(embeds[0]))"
```
What it does: Checks that each chunk is converted into a numeric vector representation.

### 4. Check question to top-5 relevant chunks
```bash
python -c "from pathlib import Path; from src.rag.pipeline import PhaseOneRAG; rag = PhaseOneRAG(Path('data/raw/self-attention-transformers-2023.pdf')); rag.pdf_to_chunks(); top = rag.question_to_top_k_chunks('What is self-attention?', top_k=5); print(len(top)); print(top[0][:500])"
```
What it does: Tests retrieval quality by finding the most relevant chunks for a question.

### 5. Check question to answer using retrieved context
```bash
python -c "from pathlib import Path; from src.rag.pipeline import PhaseOneRAG; rag = PhaseOneRAG(Path('data/raw/self-attention-transformers-2023.pdf')); answer = rag.answer_question('What is self-attention?'); print(answer[:1000])"
```
What it does: Verifies the system can answer a question using the retrieved evidence rather than free-form knowledge.

### 6. Run the phase-1 automated test
```bash
python -m pytest -q tests/test_phase1_pipeline.py
```
What it does: Runs the checkpoint-based regression test and confirms the first five stages work end-to-end.

---

## Phase 2: Data Quality and Metadata Handling

### Placeholder for future commands
```bash
# to be added after phase 2 is implemented
```
What it does: Will validate metadata extraction, document IDs, and richer ingestion logic.

---

## Phase 3: Hybrid Retrieval and Reranking

### Placeholder for future commands
```bash
# to be added after phase 3 is implemented
```
What it does: Will validate dense + sparse retrieval and reranking before generation.

---

## Phase 4: Evaluation and Tracing

### Placeholder for future commands
```bash
# to be added after phase 4 is implemented
```
What it does: Will validate retrieval metrics, answer quality, and trace logging for debugging.

---

## Phase 5: API and Docker

### Placeholder for future commands
```bash
# to be added after phase 5 is implemented
```
What it does: Will validate the FastAPI endpoints and the containerized deployment flow.

---

## Status Update

Completed phases:
- Phase 0: Environment setup
- Phase 1: PDF to answer checkpoint flow

Next phases to add:
- Phase 2: Data quality and metadata handling
- Phase 3: Hybrid retrieval and reranking
- Phase 4: Evaluation and tracing
- Phase 5: API and Docker
