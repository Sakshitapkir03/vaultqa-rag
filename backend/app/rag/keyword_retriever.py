from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Tuple

from .store import ChunkRecord


TOKEN_RE = re.compile(r"\b[a-zA-Z0-9\.\+#-]+\b")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


class KeywordRetriever:
    """
    Lightweight lexical retriever.
    Scores chunks using:
    - term overlap
    - normalized term frequency
    - phrase boost
    """

    def search(
        self,
        question: str,
        chunks: List[ChunkRecord],
        top_k: int = 10,
    ) -> List[Tuple[float, ChunkRecord]]:
        q = question.lower().strip()
        q_terms = tokenize(q)
        if not q_terms or not chunks:
            return []

        q_counts = Counter(q_terms)
        results: List[Tuple[float, ChunkRecord]] = []

        for rec in chunks:
            text = rec.text.lower()
            doc_terms = tokenize(text)
            if not doc_terms:
                continue

            d_counts = Counter(doc_terms)
            score = 0.0

            # term overlap
            overlap = sum(1 for t in q_counts if t in d_counts)
            score += overlap * 0.25

            # term frequency
            tf_score = 0.0
            for term, qf in q_counts.items():
                if term in d_counts:
                    tf = d_counts[term] / max(1, len(doc_terms))
                    tf_score += (1.0 + math.log1p(d_counts[term])) * qf * tf
            score += tf_score * 4.0

            # phrase boost
            if q in text and len(q.split()) >= 2:
                score += 1.25

            # name/entity style boost
            for term in q_terms:
                if len(term) >= 4 and term in text:
                    score += 0.03

            if score > 0:
                results.append((score, rec))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]