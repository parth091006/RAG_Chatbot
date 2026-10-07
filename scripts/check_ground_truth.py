from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.generation.generator import Generator
from src.rag.pipeline import PhaseOneRAG


ARMS = {
    "legacy-pypdf": ("legacy-pypdf", False),
    "A-current": ("current", False),
    "B-current+fix": ("current", True),
    "C-primary-only": ("primary-only", False),
    "D-primary-only+fix": ("primary-only", True),
}
DISPUTED = {"q001", "q004", "q008", "q010"}
NOISY = {"q003", "q004", "q006", "q010"}
HEADER_PATTERN = re.compile(r"with deep learning \d{1,3}(?!\w)", re.IGNORECASE)


def question_terms(text: str) -> set[str]:
    return Generator._extract_terms(text)


def overlap(question: str, text: str) -> float:
    terms = question_terms(question)
    if not terms:
        return 0.0
    return len(terms & question_terms(text)) / len(terms)


def load_questions(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_arm(pdf_path: Path, extraction: str, header_fix: bool) -> dict[str, Any]:
    pipeline = PhaseOneRAG(
        pdf_path,
        extraction=extraction,
        header_fix=header_fix,
    )
    chunks = pipeline.pdf_to_chunks()
    return {
        "pipeline": pipeline,
        "chunks": chunks,
        "by_id": {chunk["chunk_id"]: chunk for chunk in chunks},
    }


def page_map(chunks: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    by_page: dict[int, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_page.setdefault(int(chunk["page_number"]), []).append(chunk)
    for page, page_chunks in sorted(by_page.items()):
        result[str(page)] = {
            "first_chunk_id": page_chunks[0]["chunk_id"],
            "last_chunk_id": page_chunks[-1]["chunk_id"],
            "count": len(page_chunks),
        }
    return result


def git_last_change(path: Path) -> dict[str, str | None]:
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H|%cI", "--", str(path)],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "timestamp": None}
    line = result.stdout.strip()
    if not line or "|" not in line:
        return {"commit": None, "timestamp": None}
    commit, timestamp = line.split("|", 1)
    return {"commit": commit, "timestamp": timestamp}


def file_evidence(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "mtime": datetime.fromtimestamp(
            stat.st_mtime,
            tz=timezone.utc,
        ).isoformat(),
        "git_last_change": git_last_change(path),
    }


def chunk_range_ids(by_id: dict[str, dict[str, Any]], ids: list[str]) -> list[str]:
    existing = [chunk_id for chunk_id in ids if chunk_id in by_id]
    if not existing:
        return []
    numbers = [int(chunk_id.rsplit("_", 1)[-1]) for chunk_id in existing]
    prefix = existing[0].rsplit("_", 1)[0]
    return [f"{prefix}_{number:03d}" for number in range(min(numbers), max(numbers) + 1)]


def relation(set_a: list[str], set_b: list[str]) -> str:
    a = {int(value.rsplit("_", 1)[-1]) for value in set_a}
    b = {int(value.rsplit("_", 1)[-1]) for value in set_b}
    if b == {value - 1 for value in a}:
        return "uniform -1 shift"
    if a == {value - 1 for value in b}:
        return "uniform +1 shift"
    if a & b:
        return "partial overlap"
    return "no overlap"


def word_difference(current: str, primary: str) -> str:
    primary_counts = Counter(re.findall(r"\b\w+\b", primary.casefold()))
    kept: list[str] = []
    for word in re.findall(r"\b\w+\b", current):
        key = word.casefold()
        if primary_counts[key]:
            primary_counts[key] -= 1
        else:
            kept.append(word)
    return " ".join(kept)


def header_chunk_counts(chunks: list[dict[str, Any]]) -> dict[str, int]:
    starts = 0
    ends = 0
    contains = 0
    for chunk in chunks:
        text = str(chunk["text"])
        matches = list(HEADER_PATTERN.finditer(text))
        if matches:
            contains += 1
            first = text[: matches[0].start()].strip()
            last = text[matches[-1].end() :].strip()
            if not first:
                starts += 1
            if not last:
                ends += 1
    return {
        "chunks_starting_with_leaked_header": starts,
        "chunks_ending_with_leaked_header": ends,
        "chunks_containing_leaked_header": contains,
    }


def remap_chunks(
    source: dict[str, dict[str, Any]],
    target: dict[str, dict[str, Any]],
    source_name: str,
    target_name: str,
) -> dict[str, Any]:
    target_sets = {
        chunk_id: set(re.findall(r"\b\w+\b", str(chunk["text"]).casefold()))
        for chunk_id, chunk in target.items()
    }
    mappings = []
    for source_id, source_chunk in source.items():
        source_set = set(re.findall(r"\b\w+\b", str(source_chunk["text"]).casefold()))
        candidates = []
        for target_id, target_set in target_sets.items():
            union = source_set | target_set
            intersection = source_set & target_set
            jaccard = len(intersection) / len(union) if union else 1.0
            containment = len(intersection) / len(source_set) if source_set else 1.0
            candidates.append((jaccard, target_id, containment))
        candidates.sort(reverse=True)
        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else (0.0, None, 0.0)
        margin = best[0] - second[0]
        mappings.append(
            {
                "source_chunk_id": source_id,
                "target_chunk_id": best[1],
                "jaccard": round(best[0], 6),
                "containment": round(best[2], 6),
                "second_best_target_chunk_id": second[1],
                "second_best_jaccard": round(second[0], 6),
                "margin": round(margin, 6),
                "low_confidence": best[0] < 0.6 or margin < 0.1,
            }
        )
    return {
        "from": source_name,
        "to": target_name,
        "mappings": mappings,
    }


def run_report(
    pdf_path: Path,
    question_path: Path,
) -> tuple[dict[str, Any], str]:
    questions = load_questions(question_path)
    question_by_id = {item["id"]: item for item in questions}
    arms = {
        name: build_arm(pdf_path, extraction, header_fix)
        for name, (extraction, header_fix) in ARMS.items()
    }

    report: list[str] = []
    report.append("GROUND-TRUTH SAFETY REPORT")
    report.append("==========================")
    report.append("")
    report.append(f"PDF: {pdf_path}")
    report.append(f"Questions: {question_path}")
    report.append("")

    report.append("A. PER-PAGE CHUNK MAP")
    report.append("----------------------")
    maps = {}
    for name, arm in arms.items():
        maps[name] = page_map(arm["chunks"])
        report.append(name)
        for page, values in maps[name].items():
            report.append(
                f"  page {page}: {values['first_chunk_id']} -> "
                f"{values['last_chunk_id']} ({values['count']})"
            )
    report.append("")

    page_count_maps = [
        [values["count"] for _, values in sorted(mapping.items(), key=lambda item: int(item[0]))]
        for mapping in maps.values()
    ]
    same_page_mapping = all(mapping == page_count_maps[0] for mapping in page_count_maps)
    report.append(f"Global chunk IDs map to the same pages in every arm: {same_page_mapping}")
    first_divergence = None
    for page_number in range(1, 19):
        counts = {
            name: mapping.get(str(page_number), {}).get("count", 0)
            for name, mapping in maps.items()
        }
        if len(set(counts.values())) > 1:
            first_divergence = (page_number, counts)
            break
    report.append(f"First page with any chunk-count divergence: {first_divergence or 'None'}")
    report.append("")

    report.append("B. AUTHORING-BASIS EVIDENCE")
    report.append("---------------------------")
    for path in [
        question_path,
        PROJECT_ROOT / "data/evaluation/retrieval_questions_ground_truth.json",
        PROJECT_ROOT / "src/rag/ingestion/parser.py",
        PROJECT_ROOT / "src/rag/pipeline.py",
        PROJECT_ROOT / "scripts/inspect_chunks.py",
    ]:
        report.append(json.dumps(file_evidence(path), ensure_ascii=False))
    report.append("")

    report.append("C. DISPUTED QUESTIONS")
    report.append("----------------------")
    question_b_path = PROJECT_ROOT / "data/evaluation/retrieval_questions_ground_truth.json"
    questions_b = {item["id"]: item for item in load_questions(question_b_path)}
    for question_id in sorted(DISPUTED):
        item_a = question_by_id[question_id]
        item_b = questions_b[question_id]
        report.append(f"{question_id}: {item_a['question']}")
        report.append(f"  set A: {item_a['relevant_chunks']}")
        report.append(f"  set B: {item_b['relevant_chunks']}")
        report.append(f"  relation: {relation(item_a['relevant_chunks'], item_b['relevant_chunks'])}")
        if question_id in {"q004", "q010"}:
            report.append("  noisy-answer question: yes")
        ids = sorted(set(item_a["relevant_chunks"]) | set(item_b["relevant_chunks"]))
        for arm_name, arm in arms.items():
            report.append(f"  arm {arm_name}:")
            for chunk_id in chunk_range_ids(arm["by_id"], ids):
                chunk = arm["by_id"].get(chunk_id)
                if chunk is None:
                    report.append(f"    {chunk_id}: MISSING")
                    continue
                cited = []
                if chunk_id in item_a["relevant_chunks"]:
                    cited.append("A")
                if chunk_id in item_b["relevant_chunks"]:
                    cited.append("B")
                preview = " ".join(str(chunk["text"]).split())[:200]
                report.append(
                    f"    {chunk_id} cited_by={','.join(cited) or '-'} "
                    f"overlap={overlap(item_a['question'], str(chunk['text'])):.3f} "
                    f"text={preview!r}"
                )
    report.append("")

    report.append("D. UNDISPUTED QUESTIONS")
    report.append("------------------------")
    for question_id in sorted(set(question_by_id) - DISPUTED):
        item = question_by_id[question_id]
        report.append(f"{question_id}: {item['question']}")
        if question_id in NOISY:
            report.append("  noisy-answer question: yes")
        for arm_name, arm in arms.items():
            compact = []
            for chunk_id in item["relevant_chunks"]:
                chunk = arm["by_id"].get(chunk_id)
                if chunk is not None:
                    compact.append(
                        f"{chunk_id}: {' '.join(str(chunk['text']).split())[:120]!r}"
                    )
            report.append(f"  {arm_name}: " + " | ".join(compact))
    report.append("")

    report.append("E. FLAGS")
    report.append("--------")
    for arm_name, arm in arms.items():
        missing = []
        for item in questions:
            missing.extend(
                chunk_id
                for chunk_id in item["relevant_chunks"]
                if chunk_id not in arm["by_id"]
            )
        report.append(f"{arm_name} missing relevant IDs: {sorted(set(missing))}")
    report.append("Questions with no relevant chunks: []")
    ids = set.intersection(*(set(arm["by_id"]) for arm in arms.values()))
    differing_text_ids = []
    for chunk_id in sorted(ids):
        texts = {str(arm["by_id"][chunk_id]["text"]) for arm in arms.values()}
        if len(texts) > 1:
            differing_text_ids.append(chunk_id)
    report.append(f"Chunks whose text differs between arms: {len(differing_text_ids)}")
    report.append("")

    report.append("F. REMAPS")
    report.append("---------")
    report.append("Use --propose-remap with explicit --from and --to to write one pair.")
    report.append("")

    report.append("H. EVIDENCE-LOSS CHECK")
    report.append("-----------------------")
    current_pages = {
        page: text
        for page, text in enumerate(
            __import__("src.rag.ingestion.parser", fromlist=["extract_pages_from_pdf"]).extract_pages_from_pdf(pdf_path, include_secondary=True),
            start=1,
        )
    }
    primary_pages = {
        page: text
        for page, text in enumerate(
            __import__("src.rag.ingestion.parser", fromlist=["extract_pages_from_pdf"]).extract_pages_from_pdf(pdf_path, include_secondary=False),
            start=1,
        )
    }
    for item in questions:
        report.append(f"{item['id']}: {item['question']}")
        for label, ids_for_question in [("A", item["relevant_chunks"]), ("B", questions_b[item["id"]]["relevant_chunks"])]:
            pages = sorted({int(arms["A-current"]["by_id"][chunk_id]["page_number"]) for chunk_id in ids_for_question if chunk_id in arms["A-current"]["by_id"]})
            for page in pages:
                dropped = word_difference(current_pages[page], primary_pages[page])
                retained = primary_pages[page]
                dropped_overlap = overlap(item["question"], dropped)
                retained_overlap = overlap(item["question"], retained)
                higher = dropped_overlap > retained_overlap
                report.append(
                    f"  set {label} page {page}: dropped_overlap={dropped_overlap:.3f} "
                    f"retained_overlap={retained_overlap:.3f} higher={higher} "
                    f"dropped_first_300={dropped[:300]!r}"
                )
    report.append("")

    report.append("I. LEAKED HEADER CHUNK COUNTS")
    report.append("------------------------------")
    for arm_name, arm in arms.items():
        report.append(f"{arm_name}: {header_chunk_counts(arm['chunks'])}")
    report.append("")

    data = {
        "pdf": str(pdf_path),
        "questions": str(question_path),
        "arms": {
            name: {
                "extraction": extraction,
                "header_fix": header_fix,
                "chunk_count": arm["pipeline"].chunk_count,
                "total_page_characters": arm["pipeline"].total_page_characters,
                "total_chunk_characters": arm["pipeline"].total_chunk_characters,
                "page_map": maps[name],
                "header_chunk_counts": header_chunk_counts(arm["chunks"]),
            }
            for name, arm in arms.items()
            for extraction, header_fix in [(ARMS[name][0], ARMS[name][1])]
        },
        "global_chunk_ids_same_pages": same_page_mapping,
        "first_page_chunk_count_divergence": first_divergence,
    }
    return data, "\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit ground truth across extraction arms.")
    parser.add_argument("--pdf", type=Path, default=PROJECT_ROOT / "data/raw/self-attention-transformers-2023.pdf")
    parser.add_argument("--questions", type=Path, default=PROJECT_ROOT / "data/evaluation/retrieval_questions.json")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/evaluation/ground_truth_check_phase4.json")
    parser.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs/experiments/ground_truth_check_phase4.txt")
    parser.add_argument("--propose-remap", action="store_true")
    parser.add_argument("--from", dest="from_extraction", choices=list(ARMS), default=None)
    parser.add_argument("--to", dest="to_extraction", choices=list(ARMS), default=None)
    args = parser.parse_args()

    if args.propose_remap:
        if args.from_extraction is None or args.to_extraction is None:
            parser.error("--propose-remap requires explicit --from and --to")
        source = build_arm(args.pdf, *ARMS[args.from_extraction])
        target = build_arm(args.pdf, *ARMS[args.to_extraction])
        remap = remap_chunks(
            source["by_id"],
            target["by_id"],
            args.from_extraction,
            args.to_extraction,
        )
        output = PROJECT_ROOT / "data/evaluation" / f"proposed_remap_{args.from_extraction}_to_{args.to_extraction}.json"
        output.write_text(json.dumps(remap, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(output), "mappings": len(remap["mappings"])}, ensure_ascii=False))
        return

    data, report = run_report(args.pdf, args.questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(report, encoding="utf-8")
    print(report, end="")
    print(f"Saved JSON: {args.output}")
    print(f"Saved report: {args.report}")


if __name__ == "__main__":
    main()
