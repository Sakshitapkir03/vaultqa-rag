from typing import List, Tuple

from .store import ChunkRecord

def rerank_hits(question: str, hits: List[Tuple[float, ChunkRecord]], selected_source: str = ""):
    q = question.lower()
    q_terms = set(q.split())
    reranked = []

    for score, rec in hits:
        boosted = float(score)

        if selected_source and rec.source == selected_source:
            boosted += 0.12

        text_lower = rec.text.lower()
        overlap_count = sum(1 for word in q_terms if word in text_lower)
        boosted += min(0.08, overlap_count * 0.01)

        if "summary" in q or "summarize" in q or "overview" in q:
            boosted += 0.03

        reranked.append((boosted, rec))

    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked