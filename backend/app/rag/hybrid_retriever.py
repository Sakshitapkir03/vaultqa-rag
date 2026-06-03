from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from .keyword_retriever import KeywordRetriever
from .store import ChunkRecord, VectorStore


class HybridRetriever:
    """
    Dense + lexical retrieval.
    final_score = alpha * dense_norm + beta * keyword_norm
    """

    def __init__(
        self,
        embedder: SentenceTransformer,
        store: VectorStore,
        alpha: float = 0.72,
        beta: float = 0.28,
    ):
        self.embedder = embedder
        self.store = store
        self.keyword = KeywordRetriever()
        self.alpha = alpha
        self.beta = beta

    def _normalize_scores(
        self,
        hits: List[Tuple[float, ChunkRecord]],
    ) -> Dict[str, float]:
        if not hits:
            return {}

        vals = [float(s) for s, _ in hits]
        lo, hi = min(vals), max(vals)

        out: Dict[str, float] = {}
        for score, rec in hits:
            if hi - lo < 1e-9:
                out[rec.chunk_id] = 1.0
            else:
                out[rec.chunk_id] = (float(score) - lo) / (hi - lo)
        return out

    def search(
        self,
        question: str,
        filters_chunks: List[ChunkRecord] | None = None,
        dense_top_k: int = 12,
        lexical_top_k: int = 12,
        final_top_k: int = 8,
    ) -> List[Tuple[float, ChunkRecord]]:
        # dense retrieval from vector index
        q_emb = self.embedder.encode([question], show_progress_bar=False)
        dense_hits = self.store.search(np.array(q_emb), top_k=dense_top_k)

        if filters_chunks is not None:
            allowed_ids = {c.chunk_id for c in filters_chunks}
            dense_hits = [(s, r) for s, r in dense_hits if r.chunk_id in allowed_ids]
            lexical_hits = self.keyword.search(question, filters_chunks, top_k=lexical_top_k)
        else:
            lexical_hits = self.keyword.search(question, self.store.all_chunks(), top_k=lexical_top_k)

        dense_norm = self._normalize_scores(dense_hits)
        lexical_norm = self._normalize_scores(lexical_hits)

        merged: Dict[str, Dict[str, object]] = {}

        for score, rec in dense_hits:
            merged[rec.chunk_id] = {
                "rec": rec,
                "dense": dense_norm.get(rec.chunk_id, 0.0),
                "lex": 0.0,
            }

        for score, rec in lexical_hits:
            if rec.chunk_id not in merged:
                merged[rec.chunk_id] = {
                    "rec": rec,
                    "dense": 0.0,
                    "lex": lexical_norm.get(rec.chunk_id, 0.0),
                }
            else:
                merged[rec.chunk_id]["lex"] = lexical_norm.get(rec.chunk_id, 0.0)

        final_hits: List[Tuple[float, ChunkRecord]] = []
        for chunk_id, item in merged.items():
            dense_score = float(item["dense"])
            lex_score = float(item["lex"])
            rec = item["rec"]  # type: ignore[assignment]
            final_score = self.alpha * dense_score + self.beta * lex_score
            final_hits.append((final_score, rec))

        final_hits.sort(key=lambda x: x[0], reverse=True)
        return final_hits[:final_top_k]