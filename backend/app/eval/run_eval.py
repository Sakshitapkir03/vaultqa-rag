import json
import time
from pathlib import Path

from backend.app.rag.engine import RAGEngine, AskFilters
from backend.app.rag.query_intent import classify_query
from backend.app.eval.metrics import (
    keyword_coverage,
    intent_match,
    citation_source_match,
    refusal_match,
    format_cleanliness,
    brevity_score,
)

TESTSET_PATH = Path("backend/app/eval/testset.json")


def main():
    engine = RAGEngine()

    with TESTSET_PATH.open("r", encoding="utf-8") as f:
        testset = json.load(f)

    total = len(testset)
    intent_scores = []
    keyword_scores = []
    citation_scores = []
    refusal_scores = []
    cleanliness_scores = []
    brevity_scores = []
    latencies = []

    for case in testset:
        question = case["question"]
        source = case.get("source")
        expected_intent = case.get("expected_intent", "")
        expected_keywords = case.get("expected_keywords", [])
        expected_source = case.get("expected_source", source)
        should_refuse = case.get("should_refuse", False)

        predicted_intent = classify_query(question)["intent"]

        filters = AskFilters(source=source)
        t0 = time.time()
        result = engine.ask(question, filters)
        latency = time.time() - t0

        answer = result.get("answer", "")
        citations = result.get("citations", [])

        print("Top citations:")
        for c in citations[:3]:
            print(
                f"  - {c.get('source')} p{c.get('page')} | "
                f"score={c.get('score')} | "
                f"quote={c.get('quote', '')[:120]}..."
            )

        i_score = intent_match(predicted_intent, expected_intent)
        kw_score = keyword_coverage(answer, expected_keywords)
        cit_score = citation_source_match(citations, expected_source)
        ref_score = refusal_match(answer, should_refuse)
        clean_score = format_cleanliness(answer)
        brev_score = brevity_score(answer)

        intent_scores.append(i_score)
        keyword_scores.append(kw_score)
        citation_scores.append(cit_score)
        refusal_scores.append(ref_score)
        cleanliness_scores.append(clean_score)
        brevity_scores.append(brev_score)
        latencies.append(latency)

        print(f"Case: {case.get('id', 'unknown')}")
        print(f"Question: {question}")
        print(f"Expected intent: {expected_intent} | Predicted: {predicted_intent} | Match: {bool(i_score)}")
        print(f"Answer: {answer[:200]}...")
        print(f"Keyword coverage: {kw_score:.2f} | Refusal match: {ref_score:.2f} | Cleanliness: {clean_score:.2f} | Brevity: {brev_score:.2f}")
        print(f"Latency: {latency:.2f}s")
        print("-" * 60)

    summary = {
        "total_cases": total,
        "intent_accuracy":        round(sum(intent_scores)     / total, 3) if total else 0.0,
        "avg_keyword_coverage":   round(sum(keyword_scores)    / total, 3) if total else 0.0,
        "avg_citation_accuracy":  round(sum(citation_scores)   / total, 3) if total else 0.0,
        "refusal_accuracy":       round(sum(refusal_scores)    / total, 3) if total else 0.0,
        "format_cleanliness":     round(sum(cleanliness_scores)/ total, 3) if total else 0.0,
        "avg_brevity_score":      round(sum(brevity_scores)    / total, 3) if total else 0.0,
        "avg_latency_sec":        round(sum(latencies)         / total, 3) if total else 0.0,
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("SCORE TABLE  (out of 10 cases)")
    print("=" * 60)
    rows = [
        ("Intent Accuracy",       summary["intent_accuracy"]),
        ("Keyword Coverage",      summary["avg_keyword_coverage"]),
        ("Citation Accuracy",     summary["avg_citation_accuracy"]),
        ("Refusal Accuracy",      summary["refusal_accuracy"]),
        ("Format Cleanliness",    summary["format_cleanliness"]),
        ("Brevity Score",         summary["avg_brevity_score"]),
    ]
    for name, val in rows:
        pct = round(val * 100, 1)
        bar = "#" * int(pct / 5)
        print(f"  {name:<22} {pct:5.1f}%  {bar}")
    print(f"\n  Avg latency: {summary['avg_latency_sec']}s")


if __name__ == "__main__":
    main()
