from typing import List, Dict, Any, Optional


def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """Fraction of expected keywords present in the answer (case-insensitive)."""
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matched / len(expected_keywords)


def intent_match(predicted: str, expected: str) -> float:
    """Binary: 1.0 if intent classifier returned the correct label."""
    return 1.0 if predicted == expected else 0.0


def citation_source_match(citations: List[Dict[str, Any]], expected_source: str) -> float:
    """Fraction of citations that point to the expected source file."""
    if not citations:
        return 0.0
    matched = sum(1 for c in citations if c.get("source") == expected_source)
    return matched / len(citations)


def refusal_match(answer: str, should_refuse: bool) -> float:
    """1.0 if the system's refusal behaviour matches the expected behaviour."""
    refusal_markers = [
        "couldn't find enough",
        "i don't know",
        "not enough support",
        "not enough evidence",
        "could not find",
    ]
    predicted_refusal = any(marker in answer.lower() for marker in refusal_markers)
    return 1.0 if predicted_refusal == should_refuse else 0.0


def format_cleanliness(answer: str) -> float:
    """1.0 if the answer contains no prompt-leakage artefacts."""
    bad_patterns = [
        "## your task",
        "perform the following",
        "within the context",
        "instruction",
        "follow these steps",
    ]
    answer_lower = answer.lower()
    found = any(p in answer_lower for p in bad_patterns)
    return 0.0 if found else 1.0


def brevity_score(answer: str, max_chars: int = 500) -> float:
    """1.0 if the answer is within max_chars characters."""
    return 1.0 if len(answer) <= max_chars else 0.0


def recall_at_k(
    citations: List[Dict[str, Any]],
    expected_source: Optional[str],
    k: int = 5,
) -> float:
    """
    1.0 if the expected source appears in the top-k citations, else 0.0.
    Returns 1.0 (N/A) when expected_source is None (e.g., should-refuse cases).
    """
    if expected_source is None:
        return 1.0
    for citation in citations[:k]:
        if citation.get("source") == expected_source:
            return 1.0
    return 0.0


def mrr(
    citations: List[Dict[str, Any]],
    expected_source: Optional[str],
) -> float:
    """
    Mean Reciprocal Rank: 1/rank of the first citation matching expected_source.
    Returns 1.0 (N/A) when expected_source is None. Returns 0.0 if no match found.
    """
    if expected_source is None:
        return 1.0
    for rank, citation in enumerate(citations, start=1):
        if citation.get("source") == expected_source:
            return 1.0 / rank
    return 0.0