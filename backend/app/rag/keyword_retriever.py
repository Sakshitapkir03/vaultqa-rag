import re
from collections import Counter
from typing import List, Tuple

from .store import ChunkRecord

def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

def keyword_search(question: str, chunks: List[ChunkRecord], top_k: int = 10) -> List[Tuple[float, ChunkRecord]]:
    q_tokens = tokenize(question)
    q_count = Counter(q_tokens)

    scored = []
    for chunk in chunks:
        c_tokens = tokenize(chunk.text)
        c_count = Counter(c_tokens)

        overlap = sum(min(q_count[t], c_count[t]) for t in q_count)
        if overlap > 0:
            score = overlap / max(1, len(q_tokens))
            scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]