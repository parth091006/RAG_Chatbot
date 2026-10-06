from __future__ import annotations


class Comparator:
    def __init__(self):
        self._comparisons: list[dict[str, str]] = []

    def compare(self, source_a: str, source_b: str) -> dict[str, str]:
        result = {
            "source_a": source_a,
            "source_b": source_b,
            "summary": f"Comparing {source_a} against {source_b}.",
        }
        self._comparisons.append(result)
        return result
