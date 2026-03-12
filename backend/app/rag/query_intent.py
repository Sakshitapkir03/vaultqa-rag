from typing import Dict

def classify_query(question: str) -> Dict:
    q = question.lower().strip()

    if any(x in q for x in ["summarize", "summary", "overview", "main points", "key takeaways"]):
        return {"intent": "summary", "confidence": 0.92}

    if any(x in q for x in ["compare", "difference", "versus", "vs"]):
        return {"intent": "comparison", "confidence": 0.9}

    if any(x in q for x in ["when", "timeline", "chronology", "sequence"]):
        return {"intent": "timeline", "confidence": 0.84}

    if any(x in q for x in ["contradiction", "conflict", "inconsistent", "disagree"]):
        return {"intent": "contradiction_check", "confidence": 0.88}

    if any(x in q for x in ["who is", "what is", "define", "meaning of"]):
        return {"intent": "definition", "confidence": 0.8}

    if any(x in q for x in ["research", "analyze", "investigate", "synthesize"]):
        return {"intent": "research_synthesis", "confidence": 0.78}

    return {"intent": "fact_extraction", "confidence": 0.65}