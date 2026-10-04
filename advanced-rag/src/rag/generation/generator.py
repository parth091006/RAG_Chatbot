from __future__ import annotations

from .prompts import qa_prompt


class Generator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """Generate a grounded answer from a prompt."""
        return f"Grounded answer generated with model: {self.model_name}\n\n{prompt}"

    def generate_from_context(self, question: str, context: str) -> str:
        prompt = qa_prompt(context=context, question=question)
        return self.generate(prompt)
