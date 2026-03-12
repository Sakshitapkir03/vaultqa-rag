from typing import Dict, List

def build_research_plan(question: str, intent: str) -> Dict:
    q = question.strip()

    if intent == "comparison":
        subs = [
            f"What is the first subject in: {q}?",
            f"What is the second subject in: {q}?",
            f"What are the main differences related to: {q}?",
        ]
    elif intent == "timeline":
        subs = [
            f"What events are mentioned in: {q}?",
            f"What chronological sequence can be extracted from: {q}?",
        ]
    elif intent == "research_synthesis":
        subs = [
            f"What is the main topic of: {q}?",
            f"What are the major findings relevant to: {q}?",
            f"What evidence supports the findings for: {q}?",
        ]
    elif intent == "summary":
        subs = [
            f"What is the document mainly about in: {q}?",
            f"What are the most important takeaways from: {q}?",
        ]
    elif intent == "definition":
        subs = [
            q,
            f"What supporting evidence exists for: {q}?",
        ]
    else:
        subs = [
            q,
            f"What supporting evidence exists for: {q}?",
        ]

    return {
        "main_question": q,
        "intent": intent,
        "sub_questions": subs,
    }