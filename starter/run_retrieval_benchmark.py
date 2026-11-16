"""Run the three local retrieval strategies and emit auditable traces."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from baseline import hit, pack_context, validate
from convert_day6 import convert_chunk, convert_question
from retrieval import RetrievalRouter


def load_contract(chunks_path: Path, questions_path: Path):
    raw_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    raw_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    chunks = [convert_chunk(row) for row in raw_chunks]
    questions = [convert_question(row) for row in raw_questions]
    return chunks, questions


def evaluate_strategy(chunks, questions, strategy: str, top_k: int, budget: int):
    router = RetrievalRouter(chunks)
    traces = []
    for question in questions:
        started = time.perf_counter()
        results = router.search(question["question"], filters=question.get("filter"),
                                top_k=top_k, strategy=strategy)
        context, context_ids = pack_context(results, budget=budget)
        elapsed_ms = (time.perf_counter() - started) * 1000
        marked = [{**row, "hit": hit(row, question["gold"])} for row in results]
        traces.append({
            "query_id": question["id"], "split": question["split"],
            "question": question["question"], "strategy": strategy,
            "top_k": top_k, "budget_chars": budget, "filter": question.get("filter"),
            "results": marked, "context": context, "context_ids": context_ids,
            "context_chars": len(context), "elapsed_ms": elapsed_ms,
            "hit1": bool(marked and marked[0]["hit"]),
            "hit3": any(row["hit"] for row in marked[:3]),
            "context_hit": any(row["hit"] and row["id"] in context_ids for row in marked),
        })
    n = len(traces)
    return traces, {
        "strategy": strategy, "n": n,
        "hit1": sum(row["hit1"] for row in traces),
        "hit3": sum(row["hit3"] for row in traces),
        "context_hit": sum(row["context_hit"] for row in traces),
        "hit1_rate": sum(row["hit1"] for row in traces) / n if n else 0,
        "hit3_rate": sum(row["hit3"] for row in traces) / n if n else 0,
        "context_evidence_rate": sum(row["context_hit"] for row in traces) / n if n else 0,
        "avg_context_chars": statistics.mean(row["context_chars"] for row in traces) if traces else 0,
        "median_ms": statistics.median(row["elapsed_ms"] for row in traces) if traces else 0,
        "service_cost_usd": 0,
        "top_k": top_k, "context_budget_chars": budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--extra-questions", type=Path)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=6000)
    args = parser.parse_args()
    chunks, questions = load_contract(args.chunks, args.questions)
    if args.extra_questions:
        questions.extend(json.loads(args.extra_questions.read_text(encoding="utf-8")))
    selected = [row for row in questions if row.get("split") == args.split]
    # The Day 6 export has a few evidence spans split at rendered page breaks.
    # Keep strict validation as the default, but validate this corpus with the
    # explicit boundary-aware mode and record the behavior in the report.
    validate(chunks, selected, allow_page_boundary=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"split": args.split, "config": {"top_k": args.top_k, "budget_chars": args.budget}, "strategies": {}}
    for strategy in ("bm25", "dense", "hybrid"):
        traces, report = evaluate_strategy(chunks, selected, strategy, args.top_k, args.budget)
        summary["strategies"][strategy] = report
        (args.output / f"{strategy}_traces.json").write_text(
            json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
