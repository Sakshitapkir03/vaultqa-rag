"""
Hallucination evaluation harness for VaultQA.

Runs a controlled comparison:
  - BASELINE: bare llama3.1:8b with no retrieval context
  - RAG:      full VaultQA pipeline (hybrid retrieval → rerank → LLM)

An LLM-as-judge labels each answer YES (hallucinated) or NO (grounded),
using only the retrieved chunks as ground truth — not the judge's own knowledge.

Run from the backend/ directory:
    python -m app.eval.hallucination_eval

Requires: Ollama running with llama3.1:8b pulled, documents indexed.
Results saved to: backend/app/eval/hallucination_results.json
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import OllamaLLM

from app.config import settings
from app.rag.engine import AskFilters, RAGEngine
from app.rag.store import ChunkRecord

TESTSET_PATH = Path(__file__).parent / "testset.json"
RESULTS_PATH = Path(__file__).parent / "hallucination_results.json"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an impartial hallucination evaluator. Your only job is to determine whether
an answer contains claims that are NOT supported by the reference passages below.

REFERENCE PASSAGES (the only ground truth — ignore your own knowledge):
{context}

QUESTION: {question}

ANSWER TO EVALUATE:
{answer}

RULES:
1. A claim is HALLUCINATED if it states a specific fact that cannot be confirmed or
   inferred from the reference passages above.
2. A claim is NOT HALLUCINATED if every factual assertion traces to at least one passage.
3. If the answer is a refusal ("I cannot find this") it is NOT HALLUCINATED — refusals
   cannot fabricate facts.
4. Judge ONLY factual faithfulness to the passages. Ignore style, grammar, completeness.
5. Do NOT use your training knowledge — the passages are the sole source of truth.

Reply with EXACTLY one of:
  VERDICT: YES   ← answer contains at least one hallucinated claim
  VERDICT: NO    ← every factual claim is supported by the passages

Then write one sentence explaining your verdict.

VERDICT:"""

_BASELINE_PROMPT = """\
Answer the following question using only your training knowledge. Be specific.

QUESTION: {question}
ANSWER:"""


# ---------------------------------------------------------------------------
# Pure helper functions  (unit-testable — no Ollama required)
# ---------------------------------------------------------------------------

def build_judge_prompt(question: str, context: str, answer: str) -> str:
    """Return the fully-filled judge prompt string."""
    return _JUDGE_PROMPT.format(
        context=context,
        question=question,
        answer=answer,
    )


def extract_verdict(raw_output: str) -> str:
    """
    Parse judge output into "YES", "NO", or "UNKNOWN".

    Priority order:
      1. Explicit "VERDICT: YES/NO"
      2. Bare YES/NO on its own line
      3. First standalone YES/NO word anywhere
      4. Content heuristics (negation / hallucination vocabulary)
      5. "UNKNOWN" as last resort
    """
    text = raw_output.strip()

    # 1 — intended format
    m = re.search(r"VERDICT\s*:\s*(YES|NO)\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # 2 — bare YES/NO on its own line
    for line in text.splitlines():
        stripped = line.strip()
        if re.fullmatch(r"(YES|NO)", stripped, re.IGNORECASE):
            return stripped.upper()

    # 3 — first standalone word
    w = re.search(r"\b(YES|NO)\b", text, re.IGNORECASE)
    if w:
        return w.group(1).upper()

    # 4 — vocabulary heuristics
    # Check NO-specific phrases first (many contain YES-like words as substrings,
    # e.g. "not hallucinated" contains "hallucinated")
    lower = text.lower()
    no_phrases = [
        "not hallucinated", "no hallucination", "does not hallucinate",
        "is not a hallucination", "faithfully supported",
        "grounded in the", "consistent with the passages",
    ]
    if any(p in lower for p in no_phrases):
        return "NO"
    # Now check YES-specific phrases (safe: NO phrases already excluded above)
    yes_phrases = [
        "hallucinated", "fabricated",
        "not supported", "cannot be found in",
        "not present in", "not mentioned in", "not found in the",
    ]
    if any(p in lower for p in yes_phrases):
        return "YES"

    return "UNKNOWN"


def is_refusal_answer(answer: str) -> bool:
    """True when the system issued a retrieval-gated refusal instead of a fabricated answer."""
    markers = [
        "couldn't find enough",
        "could not find enough",
        "not enough strong support",
        "i couldn't find",
        "i could not find",
        "does not contain",
        "no indexed chunks",
        "no relevant",
    ]
    lower = answer.lower()
    return any(m in lower for m in markers)


def compute_reduction(
    baseline_hallucinated: int,
    rag_hallucinated: int,
    total_judged: int,
) -> Dict[str, Any]:
    """Return hallucination rates and the percentage reduction."""
    baseline_rate = baseline_hallucinated / total_judged if total_judged else 0.0
    rag_rate = rag_hallucinated / total_judged if total_judged else 0.0
    reduction_pct = (
        (baseline_rate - rag_rate) / baseline_rate * 100
        if baseline_rate > 0
        else 0.0
    )
    return {
        "baseline_hallucination_rate": round(baseline_rate, 3),
        "rag_hallucination_rate": round(rag_rate, 3),
        "hallucination_reduction_pct": round(reduction_pct, 1),
        "baseline_hallucinated_count": baseline_hallucinated,
        "rag_hallucinated_count": rag_hallucinated,
        "total_judged": total_judged,
    }


# ---------------------------------------------------------------------------
# Context reconstruction
# ---------------------------------------------------------------------------

def _get_context_for_judge(
    engine: RAGEngine,
    question: str,
    source: Optional[str],
    n_chunks: int = 8,
) -> str:
    """
    Re-run the retrieval pipeline (without the LLM step) to get the exact
    context that was (or would have been) passed to the model.
    This gives the judge the right ground truth: not the whole document,
    but specifically what the RAG system retrieved.
    """
    if source:
        chunks: List[ChunkRecord] = engine.store.get_chunks_by_source(source)
    else:
        chunks = engine.store.all_chunks()

    if not chunks:
        return "No context retrieved."

    query_filters = engine._extract_query_filters(question)
    filtered = engine._filter_chunks_by_query_metadata(chunks, query_filters)
    if not filtered:
        filtered = chunks

    q_lower = question.lower()
    summary_mode = any(p in q_lower for p in [
        "summarize", "summary", "main points", "key takeaways", "overview",
    ])

    hits = engine._search_candidate_hits(question, filtered, summary_mode)
    if not hits:
        return "No context retrieved."

    expanded = engine._expand_hits_with_neighbors(hits, filtered, question)
    final_hits = (expanded if expanded else hits)[:n_chunks]

    return "\n\n".join(
        f"[{rec.source} p{rec.page}]\n{rec.text}"
        for _, rec in final_hits
    )


# ---------------------------------------------------------------------------
# LLM factories
# ---------------------------------------------------------------------------

def _make_baseline_llm() -> OllamaLLM:
    """Bare LLM — same model as RAG, no retrieval context.
    num_ctx=2048 keeps memory usage low on M-series Macs with limited unified memory."""
    return OllamaLLM(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_URL,
        temperature=0.05,
        num_predict=400,
        num_ctx=2048,
        top_k=15,
        top_p=0.85,
    )


def _make_judge_llm() -> OllamaLLM:
    """Judge LLM — temperature=0 for deterministic verdicts.
    num_ctx=3072 fits the judge prompt (context ~500 + answer ~300 tokens) comfortably."""
    return OllamaLLM(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_URL,
        temperature=0.0,
        num_predict=200,
        num_ctx=3072,
        top_k=10,
        top_p=0.9,
    )


def _make_eval_rag_llm() -> OllamaLLM:
    """RAG LLM override for eval — smaller context window than production (8192)
    to avoid OOM on Macs with 8–16 GB unified memory."""
    return OllamaLLM(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_URL,
        temperature=0.05,
        num_predict=500,
        num_ctx=3072,
        top_k=15,
        top_p=0.85,
    )


# ---------------------------------------------------------------------------
# Core eval logic
# ---------------------------------------------------------------------------

def _judge(
    judge_llm: OllamaLLM,
    question: str,
    context: str,
    answer: str,
) -> Tuple[str, str]:
    """Run the judge and return (verdict, raw_output)."""
    prompt = build_judge_prompt(question, context, answer)
    raw = (judge_llm.invoke(prompt) or "").strip()
    return extract_verdict(raw), raw


def run_hallucination_eval() -> Dict[str, Any]:
    print("Loading RAG engine and models (this takes ~30s for model warm-up)…")
    engine = RAGEngine()

    # Override engine.llm with a memory-safe context window for eval.
    # Production uses num_ctx=8192; that OOMs on M-series Macs during multi-LLM evals.
    # The eval prompts fit comfortably in 3072 tokens, so no quality loss.
    engine.llm = _make_eval_rag_llm()

    baseline_llm = _make_baseline_llm()
    judge_llm = _make_judge_llm()

    with TESTSET_PATH.open("r", encoding="utf-8") as f:
        testset = json.load(f)

    refusal_cases = [c for c in testset if c.get("should_refuse")]
    eval_cases    = [c for c in testset if not c.get("should_refuse")]

    per_case: List[Dict[str, Any]] = []
    baseline_hallucinated = 0
    rag_hallucinated = 0
    unknown_count = 0

    print(f"\n{'='*62}")
    print(f"Evaluating {len(eval_cases)} cases for hallucination…")
    print(f"Refusal cases ({len(refusal_cases)}) tracked separately.")
    print(f"{'='*62}\n")

    for case in eval_cases:
        question  = case["question"]
        source    = case.get("source")
        case_id   = case.get("id", "?")

        print(f"[{case_id}]  {question[:72]}")

        # 1 — Baseline answer (no retrieval context)
        t0 = time.perf_counter()
        baseline_ans = (
            baseline_llm.invoke(_BASELINE_PROMPT.format(question=question)) or ""
        ).strip()
        baseline_t = time.perf_counter() - t0
        print(f"  BASELINE ({baseline_t:.1f}s): {baseline_ans[:90]}…")

        # 2 — RAG answer
        t0 = time.perf_counter()
        rag_result  = engine.ask(question, AskFilters(source=source))
        rag_ans     = rag_result.get("answer", "")
        rag_t       = time.perf_counter() - t0
        print(f"  RAG      ({rag_t:.1f}s): {rag_ans[:90]}…")

        # 3 — Context (what the RAG system actually retrieved — ground truth for judge)
        context = _get_context_for_judge(engine, question, source)

        # 4 — Judge baseline
        base_verdict, base_raw = _judge(judge_llm, question, context, baseline_ans)

        # 5 — Judge RAG  (refusals are automatically NOT hallucinated)
        rag_is_refusal = is_refusal_answer(rag_ans)
        if rag_is_refusal:
            rag_verdict, rag_raw = "NO", "(auto-NO: refusal answer cannot hallucinate)"
        else:
            rag_verdict, rag_raw = _judge(judge_llm, question, context, rag_ans)

        # Tally
        if base_verdict == "YES":
            baseline_hallucinated += 1
        if rag_verdict == "YES":
            rag_hallucinated += 1
        if base_verdict == "UNKNOWN" or rag_verdict == "UNKNOWN":
            unknown_count += 1

        print(f"  Judge → BASELINE: {base_verdict}  |  RAG: {rag_verdict}\n")

        per_case.append({
            "id": case_id,
            "question": question,
            "source": source,
            "baseline_answer": baseline_ans,
            "rag_answer": rag_ans,
            "rag_is_refusal": rag_is_refusal,
            "context_preview": context[:400] + "…" if len(context) > 400 else context,
            "baseline_verdict": base_verdict,
            "rag_verdict": rag_verdict,
            "baseline_judge_raw": base_raw,
            "rag_judge_raw": rag_raw,
            "baseline_latency_sec": round(baseline_t, 2),
            "rag_latency_sec": round(rag_t, 2),
        })

    # Refusal accuracy
    refusal_per_case: List[Dict[str, Any]] = []
    refusal_correct = 0
    print(f"\nChecking {len(refusal_cases)} refusal cases…")
    for case in refusal_cases:
        question = case["question"]
        result   = engine.ask(question, AskFilters(source=case.get("source")))
        rag_ans  = result.get("answer", "")
        correct  = is_refusal_answer(rag_ans)
        if correct:
            refusal_correct += 1
        print(f"  [{case.get('id')}] refused correctly: {correct}")
        refusal_per_case.append({
            "id": case.get("id"),
            "question": question,
            "rag_answer": rag_ans,
            "correctly_refused": correct,
        })

    # Compute summary
    total_judged = len(eval_cases)
    summary = compute_reduction(baseline_hallucinated, rag_hallucinated, total_judged)
    summary["unknown_verdict_count"] = unknown_count
    summary["refusal_accuracy"] = (
        round(refusal_correct / len(refusal_cases), 3) if refusal_cases else 1.0
    )

    output = {
        "summary": summary,
        "eval_cases": per_case,
        "refusal_cases": refusal_per_case,
        "metadata": {
            "model": settings.OLLAMA_MODEL,
            "judge_model": settings.OLLAMA_MODEL,
            "judge_context_strategy": "retrieved_chunks_only",
            "eval_case_count": total_judged,
            "refusal_case_count": len(refusal_cases),
            "note": (
                "Judge uses same model as generator — known limitation. "
                "Famous-fact cases (Kafka plot, CS fundamentals) are included "
                "without exclusion, making the reduction estimate conservative."
            ),
        },
    }

    RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    _print_summary(summary)
    print(f"\nFull results saved to {RESULTS_PATH}")
    return output


def _print_summary(summary: Dict[str, Any]) -> None:
    b_rate = summary["baseline_hallucination_rate"] * 100
    r_rate = summary["rag_hallucination_rate"] * 100
    reduc  = summary["hallucination_reduction_pct"]
    ref_acc = summary["refusal_accuracy"] * 100

    print(f"\n{'='*62}")
    print("HALLUCINATION EVALUATION SUMMARY")
    print(f"{'='*62}")
    print(f"  Cases judged (non-refusal):   {summary['total_judged']}")
    print(f"  Baseline hallucinated:         {summary['baseline_hallucinated_count']} / {summary['total_judged']}  ({b_rate:.1f}%)")
    print(f"  RAG hallucinated:              {summary['rag_hallucinated_count']} / {summary['total_judged']}  ({r_rate:.1f}%)")
    print(f"  Hallucination reduction:       {reduc:.1f}%")
    print(f"  Refusal accuracy:              {ref_acc:.1f}%")
    print(f"  Unknown/unparsed verdicts:     {summary['unknown_verdict_count']}")
    print(f"{'='*62}")
    print(f"\n  ★  VaultQA reduces hallucinations by {reduc:.1f}% vs. bare LLM")
    print(f"     Baseline: {b_rate:.0f}% hallucination rate  →  RAG: {r_rate:.0f}% hallucination rate")
    print(f"{'='*62}")


if __name__ == "__main__":
    run_hallucination_eval()
