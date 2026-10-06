from __future__ import annotations


def qa_prompt(context: str, question: str) -> str:
    return f"""
You answer using only the provided context.

Context:
{context}

Question:
{question}

Requirements:
- use only the supplied evidence
- say when information is missing
- do not invent facts
- include citations
""".strip()


def comparison_prompt(context: str, question: str) -> str:
    return f"""
Compare the sources using only the evidence provided.

Context:
{context}

Question:
{question}

Requirements:
1. identify evidence for each source
2. mention similarities and differences
3. avoid unsupported inference
4. attach citations
""".strip()
