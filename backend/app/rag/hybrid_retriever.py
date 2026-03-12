from typing import List, Tuple, Dict
import numpy as np

from .store import ChunkRecord, VectorStore
from .keyword_retriever import keyword_search

def hybrid_retrieve(
    question: str,
    query_emb: np.ndarray,
    store: VectorStore,
    all_chunks: List[ChunkRecord],
    top_k: int = 12,
):
    semantic_hits = store.search(query_emb, top_k=top_k)
    keyword_hits = keyword_search(question, all_chunks, top_k=top_k)

    merged: Dict[str, Tuple[float, ChunkRecord]] = {}

    for score, rec in semantic_hits:
        merged[rec.chunk_id] = (0.7 * float(score), rec)

    for score, rec in keyword_hits:
        if rec.chunk_id in merged:
            prev, _ = merged[rec.chunk_id]
            merged[rec.chunk_id] = (prev + 0.3 * float(score), rec)
        else:
            merged[rec.chunk_id] = (0.3 * float(score), rec)

    results = list(merged.values())
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]