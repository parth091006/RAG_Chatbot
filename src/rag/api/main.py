from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="Advanced RAG Research Assistant")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query")
def query(question: str) -> dict[str, str]:
    return {"answer": f"Answer for: {question}", "citations": []}
