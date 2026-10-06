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
python -c "from pathlib import Path; from src.rag.pipeline import PhaseOneRAG; rag = PhaseOneRAG(Path('data/raw/self-attention-transformers-2023.pdf')); chunks = rag.pdf_to_chunks(); print(len(chunks)); print(chunks[0]['chunk_id']); print(chunks[0]['text'][:300])"
```
What it does: Verifies the extracted text is split into manageable chunks for retrieval.

### 3. Check chunk to embedding conversion
```bash
python -c "from pathlib import Path; from src.rag.pipeline import PhaseOneRAG; rag = PhaseOneRAG(Path('data/raw/self-attention-transformers-2023.pdf')); rag.pdf_to_chunks(); embeds = rag.chunks_to_embeddings(rag.chunks); print(len(embeds)); print(len(embeds[0]))"
```
What it does: Checks that each chunk is converted into a numeric vector representation.

### 4. Check question to top-5 relevant chunks
```bash
python -c "from pathlib import Path; from src.rag.pipeline import PhaseOneRAG; rag = PhaseOneRAG(Path('data/raw/self-attention-transformers-2023.pdf')); rag.pdf_to_chunks(); top = rag.question_to_top_k_chunks('What is self-attention?', top_k=5); print(len(top)); print(top[0]['chunk_id'], top[0]['page_number']); print(top[0]['text'][:500])"
```
What it does: Tests retrieval quality by finding the most relevant chunks for a question.

### 5. Check question to answer using retrieved context
```bash
python -c "from pathlib import Path; from src.rag.pipeline import PhaseOneRAG; rag = PhaseOneRAG(Path('data/raw/self-attention-transformers-2023.pdf')); result = rag.answer_question('What is self-attention?'); print(result['answer']); print(result['sources'])"
```
What it does: Verifies the system returns a clean final answer and citation sources without exposing the internal prompt or retrieved context construction.

### 6. Run the phase-1 automated test
```bash
python -m pytest -q tests/test_phase1_pipeline.py
```
What it does: Runs the checkpoint-based regression test and confirms the first five stages work end-to-end.

### 7. Evaluate ten questions with chunks and scores
```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py
```
What it does: Runs all questions in `data/evaluation/retrieval_questions.json`, retrieves the top five chunks for each question, prints their cosine similarity scores, and saves the complete report to `data/evaluation/retrieval_results.json`.

Before using retrieval metrics, fill each question's `relevant_chunks` list with the gold chunk IDs for the selected stable chunking configuration. The evaluator then calculates Recall@1, Recall@3, Recall@5, and MRR.

To use a different question file, PDF, or number of results:
```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py --questions data/evaluation/retrieval_questions.json --pdf data/raw/self-attention-transformers-2023.pdf --top-k 5 --output data/evaluation/retrieval_results.json
```
What it does: Runs the same evaluation with explicit input and output paths, making the test repeatable after future pipeline changes.

### 8. Verify metadata-aware chunks
```powershell
python -c "from pathlib import Path; from src.rag.pipeline import PhaseOneRAG; rag = PhaseOneRAG(Path('data/raw/self-attention-transformers-2023.pdf')); chunks = rag.pdf_to_chunks(); rag.chunks_to_embeddings(); print(chunks[0]); print(rag.question_to_scored_chunks('What is self-attention?', top_k=1)[0])"
```
What it does: Confirms each chunk keeps its `chunk_id`, document name, page number, text, embedding, and retrieval score together for future citations.

### 9. Compare chunking strategies
```powershell
.\.venv\Scripts\python.exe scripts\run_chunking_experiment.py
```
What it does: Compares the current baseline, smaller chunks, and section-aware chunks, recording chunk count, average size, and Recall@5 in `data/evaluation/chunking_experiment_results.json`. Recall remains `null` until `relevant_chunks` is annotated.

Current experiment output:

| Approach | Chunks | Average words | Estimated tokens | Recall@5 |
| --- | ---: | ---: | ---: | ---: |
| Current baseline | 38 | 214.55 | 336.08 | null until annotated |
| Smaller chunks | 65 | 131.74 | 204.95 | null until annotated |
| Section-aware chunks | 44 | 183.02 | 286.45 | null until annotated |

Recall@5 is calculated from hand-labeled `relevant_chunks`; chunk IDs must be re-annotated if the chunking strategy changes.

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
- Phase 1 retrieval evaluation: 10 questions with top-5 chunks and similarity scores
- Phase 1 metadata: page-aware chunk IDs and embedding associations
- Chunking experiment: baseline, smaller, and section-aware comparison

Next phases to add:
- Phase 2: Data quality and metadata handling
- Phase 3: Hybrid retrieval and reranking
- Phase 4: Evaluation and tracing
- Phase 5: API and Docker