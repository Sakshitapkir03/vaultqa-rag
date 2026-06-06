"""
End-to-end evaluation harness for VaultQA.

Run from the backend/ directory:
    python -m app.eval.run_eval

Requires: Ollama running with llama3.1:8b pulled, and documents indexed.
Results are saved to backend/app/eval/eval_results.json.
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from app.rag.engine import RAGEngine, AskFilters
from app.rag.query_intent import classify_query
from app.eval.metrics import (
    keyword_coverage,
    intent_match,
    citation_source_match,
    refusal_match,
    format_cleanliness,
    brevity_score,
    recall_at_k,
    mrr,
)

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_PATH = Path(__file__).parent / "eval_results.json"

RECALL_K = 5


def run_eval() -> dict:
    engine = RAGEngine()

    with TESTSET_PATH.open("r", encoding="utf-8") as f:
        testset = json.load(f)

    total = len(testset)
    per_case = []

    intent_scores: list[float] = []
    keyword_scores: list[float] = []
    citation_scores: list[float] = []
    refusal_scores: list[float] = []
    cleanliness_scores: list[float] = []
    brevity_scores: list[float] = []
    recall_scores: list[float] = []
    mrr_scores: list[float] = []
    latencies: list[float] = []

    for case in testset:
        question = case["question"]
        source = case.get("source")
        expected_intent = case.get("expected_intent", "")
        expected_keywords = case.get("expected_keywords", [])
        expected_source = case.get("expected_source", source)
        should_refuse = case.get("should_refuse", False)

        predicted_intent = classify_query(question)["intent"]

        filters = AskFilters(source=source)
        t0 = time.perf_counter()
        result = engine.ask(question, filters)
        latency = time.perf_counter() - t0

        answer = result.get("answer", "")
        citations = result.get("citations", [])

        i_score = intent_match(predicted_intent, expected_intent)
        kw_score = keyword_coverage(answer, expected_keywords)
        cit_score = citation_source_match(citations, expected_source)
        ref_score = refusal_match(answer, should_refuse)
        clean_score = format_cleanliness(answer)
        brev_score = brevity_score(answer)
        rec_score = recall_at_k(citations, expected_source, k=RECALL_K)
        mrr_score = mrr(citations, expected_source)

        intent_scores.append(i_score)
        keyword_scores.append(kw_score)
        citation_scores.append(cit_score)
        refusal_scores.append(ref_score)
        cleanliness_scores.append(clean_score)
        brevity_scores.append(brev_score)
        recall_scores.append(rec_score)
        mrr_scores.append(mrr_score)
        latencies.append(latency)

        case_result = {
            "id": case.get("id", "unknown"),
            "question": question,
            "expected_intent": expected_intent,
            "predicted_intent": predicted_intent,
            "intent_match": bool(i_score),
            "keyword_coverage": round(kw_score, 3),
            "citation_accuracy": round(cit_score, 3),
            "refusal_match": bool(ref_score),
            "format_cleanliness": bool(clean_score),
            f"recall@{RECALL_K}": round(rec_score, 3),
            "mrr": round(mrr_score, 3),
            "latency_sec": round(latency, 3),
            "top_citations": [
                {
                    "source": c.get("source"),
                    "page": c.get("page"),
                    "score": c.get("score"),
                    "quote": c.get("quote", "")[:120],
                }
                for c in citations[:3]
            ],
        }
        per_case.append(case_result)

        print(f"\n[{case.get('id')}] {question[:80]}")
        print(f"  Intent: {expected_intent} → {predicted_intent}  {'✓' if i_score else '✗'}")
        print(f"  Keywords: {kw_score:.2f}  | Recall@{RECALL_K}: {rec_score:.2f}  | MRR: {mrr_score:.2f}")
        print(f"  Refusal: {'✓' if ref_score else '✗'}  | Clean: {'✓' if clean_score else '✗'}  | Latency: {latency:.2f}s")

    summary = {
        "total_cases": total,
        "intent_accuracy":        round(sum(intent_scores)      / total, 3),
        "avg_keyword_coverage":   round(sum(keyword_scores)     / total, 3),
        "avg_citation_accuracy":  round(sum(citation_scores)    / total, 3),
        "refusal_accuracy":       round(sum(refusal_scores)     / total, 3),
        "format_cleanliness":     round(sum(cleanliness_scores) / total, 3),
        "avg_brevity_score":      round(sum(brevity_scores)     / total, 3),
        f"recall@{RECALL_K}":     round(sum(recall_scores)      / total, 3),
        "mrr":                    round(sum(mrr_scores)         / total, 3),
        "avg_latency_sec":        round(statistics.mean(latencies), 3),
        "median_latency_sec":     round(statistics.median(latencies), 3),
        "p90_latency_sec":        round(sorted(latencies)[int(0.90 * len(latencies))], 3),
    }

    output = {"summary": summary, "per_case": per_case}

    RESULTS_PATH.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    _print_summary(summary)
    print(f"\nResults saved to {RESULTS_PATH}")
    return output


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 62)
    print("EVALUATION SUMMARY")
    print("=" * 62)
    rows = [
        ("Intent Accuracy",      summary["intent_accuracy"]),
        ("Keyword Coverage",     summary["avg_keyword_coverage"]),
        ("Citation Accuracy",    summary["avg_citation_accuracy"]),
        ("Refusal Accuracy",     summary["refusal_accuracy"]),
        ("Format Cleanliness",   summary["format_cleanliness"]),
        (f"Recall@{RECALL_K}",   summary[f"recall@{RECALL_K}"]),
        ("MRR",                  summary["mrr"]),
    ]
    for name, val in rows:
        pct = round(val * 100, 1)
        bar = "█" * int(pct / 5)
        print(f"  {name:<22} {pct:5.1f}%  {bar}")

    print(f"\n  Avg latency:    {summary['avg_latency_sec']}s")
    print(f"  Median latency: {summary['median_latency_sec']}s")
    print(f"  P90 latency:    {summary['p90_latency_sec']}s")
    print("=" * 62)


if __name__ == "__main__":
    run_eval()
