from typing import Dict, List

def verify_answer(answer: str, citations: List[Dict]) -> Dict:
    grounded = len(citations) > 0 and bool(answer.strip())
    return {
        "verified": grounded,
        "citation_count": len(citations),
    }