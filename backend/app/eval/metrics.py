from typing import List, Dict, Any


def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matched / len(expected_keywords)


def intent_match(predicted: str, expected: str) -> float:
    return 1.0 if predicted == expected else 0.0


def citation_source_match(citations: List[Dict[str, Any]], expected_source: str) -> float:
    if not citations:
        return 0.0
    matched = sum(1 for c in citations if c.get("source") == expected_source)
    return matched / len(citations)


def refusal_match(answer: str, should_refuse: bool) -> float:
    refusal_markers = [
        "couldn’t find enough",
        "couldn't find enough",
        "i don't know",
        "not enough support",
        "not enough evidence",
        "could not find",
    ]
    predicted_refusal = any(marker in answer.lower() for marker in refusal_markers)
    return 1.0 if predicted_refusal == should_refuse else 0.0

def format_cleanliness(answer: str) -> float:
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
    return 1.0 if len(answer) <= max_chars else 0.0