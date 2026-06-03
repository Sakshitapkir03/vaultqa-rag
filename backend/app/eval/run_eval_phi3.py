"""
Eval runner that uses phi3:mini + a capped context window (4096 tokens)
to avoid GPU OOM on M3 with llama3.1:8b.  All metrics are identical to
run_eval.py; the only difference is the model override and reduced
retrieval limits so the generated prompt stays well under 4096 tokens.
"""

import json
import os
import time
from pathlib import Path

# Override before any backend imports resolve settings
os.environ.setdefault("OLLAMA_MODEL", "phi3:mini")

# Patch settings object after env var is set
from backend.app.config import settings  # noqa: E402
object.__setattr__(settings, "OLLAMA_MODEL", "phi3:mini")

from backend.app.rag.engine import RAGEngine, AskFilters  # noqa: E402
from backend.app.rag.query_intent import classify_query  # noqa: E402
from backend.app.eval.metrics import (  # noqa: E402
    keyword_coverage,
    intent_match,
    citation_source_match,
    refusal_match,
    format_cleanliness,
    brevity_score,
)

TESTSET_PATH = Path("backend/app/eval/testset.json")

# -------------------------------------------------------------------
# Monkey-patch retrieval limits to stay within phi3:mini's 4k window.
# Summary/chapter queries were pulling 20-25 large chunks; cap at 6.
# -------------------------------------------------------------------
from backend.app.rag import engine as _eng_mod  # noqa: E402

_orig_search = _eng_mod.RAGEngine._search_candidate_hits


def _capped_search(self, question, chunks, summary_mode):
    hits = _orig_search(self, question, chunks, summary_mode)
    return hits[:6]


_eng_mod.RAGEngine._search_candidate_hits = _capped_search

_orig_expand = _eng_mod.RAGEngine._expand_hits_with_neighbors


def _capped_expand(self, hits, source_chunks, question="", neighbors=1, max_expanded=15):
    return _orig_expand(self, hits, source_chunks, question, neighbors=1, max_expanded=6)


_eng_mod.RAGEngine._expand_hits_with_neighbors = _capped_expand


def main():
    engine = RAGEngine()
    # Lower the Ollama context window to match phi3:mini's sweet spot
    engine.llm.num_ctx = 4096
    engine.llm.num_predict = 400

    with TESTSET_PATH.open("r", encoding="utf-8") as f:
        testset = json.load(f)

    total = len(testset)
    intent_scores, keyword_scores, citation_scores = [], [], []
    refusal_scores, cleanliness_scores, brevity_scores, latencies = [], [], [], []

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
                f"quote={c.get('quote', '')[:80]}..."
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

        print(f"\nCase: {case.get('id', 'unknown')}")
        print(f"Question: {question}")
        print(f"Expected intent: {expected_intent} | Predicted: {predicted_intent} | Match: {bool(i_score)}")
        print(f"Answer: {answer[:300]}...")
        print(f"Keyword coverage: {kw_score:.2f} | Refusal match: {ref_score:.2f} | "
              f"Cleanliness: {clean_score:.2f} | Brevity: {brev_score:.2f}")
        print(f"Latency: {latency:.2f}s")
        print("-" * 60)

    summary = {
        "total_cases": total,
        "intent_accuracy":       round(sum(intent_scores)      / total, 3),
        "avg_keyword_coverage":  round(sum(keyword_scores)     / total, 3),
        "avg_citation_accuracy": round(sum(citation_scores)    / total, 3),
        "refusal_accuracy":      round(sum(refusal_scores)     / total, 3),
        "format_cleanliness":    round(sum(cleanliness_scores) / total, 3),
        "avg_brevity_score":     round(sum(brevity_scores)     / total, 3),
        "avg_latency_sec":       round(sum(latencies)          / total, 3),
    }

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 60)
    print("SCORE TABLE  (out of 10 cases)")
    print("=" * 60)
    rows = [
        ("Intent Accuracy",    summary["intent_accuracy"]),
        ("Keyword Coverage",   summary["avg_keyword_coverage"]),
        ("Citation Accuracy",  summary["avg_citation_accuracy"]),
        ("Refusal Accuracy",   summary["refusal_accuracy"]),
        ("Format Cleanliness", summary["format_cleanliness"]),
        ("Brevity Score",      summary["avg_brevity_score"]),
    ]
    for name, val in rows:
        pct = round(val * 100, 1)
        bar = "#" * int(pct / 5)
        print(f"  {name:<22} {pct:5.1f}%  {bar}")
    print(f"\n  Avg latency: {summary['avg_latency_sec']}s")


if __name__ == "__main__":
    main()
