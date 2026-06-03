from __future__ import annotations

from typing import List, Tuple

from .store import ChunkRecord


class SimpleReranker:
    """
    Lightweight reranker using query-aware heuristics.
    Useful when you do not want a heavy cross-encoder.
    """

    def rerank(
        self,
        question: str,
        hits: List[Tuple[float, ChunkRecord]],
        top_k: int = 4,
    ) -> List[Tuple[float, ChunkRecord]]:
        q = question.lower().strip()
        q_terms = set(q.replace("?", "").split())

        reranked: List[Tuple[float, ChunkRecord]] = []

        for base_score, rec in hits:
            text = rec.text.lower()
            score = float(base_score)

            # exact name/entity overlap
            overlap = sum(1 for t in q_terms if len(t) >= 3 and t in text)
            score += overlap * 0.03

            # strong boost for explicit definition phrasing
            if q.startswith("who is") or q.startswith("what is"):
                if " is " in text or " was " in text:
                    score += 0.10

            # strong boost for chapter/section queries
            if "chapter" in q or "section" in q or "part " in q:
                chapter_patterns = ["chapter ", "section ", "part ", "chapter:", "section:"]
                if any(pat in text for pat in chapter_patterns):
                    score += 0.15
                # Also boost if text starts with heading-like structure
                if text[:30].isupper() or any(text.startswith(f) for f in ["chapter ", "section "]):
                    score += 0.08

            # summary question: prefer richer chunks
            if "summarize" in q or "summary" in q or "overview" in q:
                score += min(len(rec.text) / 1500.0, 0.12)

            # slight penalty for metadata-ish chunks (lowered from -0.10)
            metadata_patterns = [
                "genre:",
                "genre tags:",
                "table of contents",
                "references",
            ]
            for pat in metadata_patterns:
                if pat in text:
                    score -= 0.05

            reranked.append((score, rec))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return reranked[:top_k]