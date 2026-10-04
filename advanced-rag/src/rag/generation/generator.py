from __future__ import annotations


class Generator:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name

    def generate(self, prompt: str) -> str:
        """Generate a grounded answer from a prompt. This placeholder returns the prompt text."""
        return f"Grounded answer generated with model: {self.model_name}\n\n{prompt}"
