from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from ..config import settings
from .chunking import chunk_text, guess_doc_type
from .pdf_extract import extract_pdf_pages
from .store import VectorStore, ChunkRecord, UPLOAD_DIR, register_doc


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def prompt_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


@dataclass
class AskFilters:
    doc_type: Optional[str] = None
    collection: Optional[str] = None
    source: Optional[str] = None


class RAGEngine:
    def __init__(self):
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.store = VectorStore(dim=self.dim)

    def ingest_file(self, filename: str, collection: str, doc_type: Optional[str]) -> Dict[str, Any]:
        path = UPLOAD_DIR / filename
        if not path.exists():
            return {"status": "error", "message": "file not found"}

        inferred_type = doc_type or guess_doc_type(filename)
        t0 = time.time()

        records: List[ChunkRecord] = []
        texts: List[str] = []

        if filename.lower().endswith(".pdf"):
            pages = extract_pdf_pages(str(path))
            for page_num, page_text in pages:
                chunks = chunk_text(page_text)
                for ci, ch in enumerate(chunks):
                    chunk_id = f"{filename}::p{page_num}::c{ci}"
                    records.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            source=filename,
                            page=page_num,
                            doc_type=inferred_type,
                            collection=collection,
                            text=ch,
                        )
                    )
                    texts.append(ch)
        else:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            chunks = chunk_text(raw)
            for ci, ch in enumerate(chunks):
                chunk_id = f"{filename}::p0::c{ci}"
                records.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        source=filename,
                        page=0,
                        doc_type=inferred_type,
                        collection=collection,
                        text=ch,
                    )
                )
                texts.append(ch)

        if not texts:
            return {"status": "error", "message": "no extractable text found"}

        embs = self.embedder.encode(texts, batch_size=32, show_progress_bar=False)
        self.store.add_embeddings(np.array(embs), records)

        register_doc(
            {
                "filename": filename,
                "collection": collection,
                "doc_type": inferred_type,
                "indexed_chunks": len(records),
                "indexed_at": int(time.time()),
            }
        )

        return {
            "status": "ok",
            "filename": filename,
            "collection": collection,
            "doc_type": inferred_type,
            "chunks": len(records),
            "seconds": round(time.time() - t0, 3),
        }

    def _apply_filters(
        self, hits: List[Tuple[float, ChunkRecord]], f: AskFilters
    ) -> List[Tuple[float, ChunkRecord]]:
        out = []
        for score, rec in hits:
            if f.doc_type and rec.doc_type != f.doc_type:
                continue
            if f.collection and rec.collection != f.collection:
                continue
            if f.source and rec.source != f.source:
                continue
            out.append((score, rec))
        return out

    def ask(self, question: str, filters: AskFilters, top_k: int = 8) -> Dict[str, Any]:
        t0 = time.time()
        q_lower = question.lower().strip()

        summary_mode = any(
            phrase in q_lower
            for phrase in [
                "summarize",
                "summary",
                "main points",
                "key takeaways",
                "overview",
                "what is this document about",
                "explain this document",
            ]
        )

        source_mode = bool(filters.source)

        # If a specific document is selected, directly use chunks from that doc.
        if source_mode:
            source_chunks = self.store.get_chunks_by_source(filters.source)
            if not source_chunks:
                return {
                    "answer": "I couldn't find any indexed chunks for the selected document.",
                    "grounded": False,
                    "confidence": 0.0,
                    "citations": [],
                    "contradiction": False,
                    "audit": {
                        "source_mode": True,
                        "filters": filters.__dict__,
                        "model": settings.OLLAMA_MODEL,
                        "latency_s": round(time.time() - t0, 3),
                    },
                }

            picked = source_chunks[: 5 if summary_mode else 4]

            citations = []
            for rec in picked:
                citations.append(
                    {
                        "source": rec.source,
                        "page": rec.page,
                        "chunk_id": rec.chunk_id,
                        "score": 0.2,
                        "quote": rec.text[:320] + ("..." if len(rec.text) > 320 else ""),
                    }
                )

            context = "\n\n".join(
                [f"[{c['source']} p{c['page']}]\n{c['quote']}" for c in citations]
            )

            prompt = (
                "You are VaultQA, a document-only assistant.\n"
                "Use ONLY the provided context.\n"
                "Do not use outside knowledge.\n"
                "Answer clearly and faithfully from the document.\n"
                "If this is a summary-style question, summarize the selected document.\n"
                "At the end, include short source references like (source p#).\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {question}\n"
                "ANSWER:"
            )

            import requests

            r = requests.post(
                f"{settings.OLLAMA_URL}/api/generate",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=120,
            )
            r.raise_for_status()
            answer = r.json().get("response", "").strip()

            return {
                "answer": answer,
                "grounded": True,
                "confidence": 0.82,
                "citations": citations[:4],
                "contradiction": False,
                "audit": {
                    "source_mode": True,
                    "summary_mode": summary_mode,
                    "filters": filters.__dict__,
                    "model": settings.OLLAMA_MODEL,
                    "latency_s": round(time.time() - t0, 3),
                },
            }

        # Fallback: semantic retrieval across all docs
        q_emb = self.embedder.encode([question], show_progress_bar=False)
        hits = self.store.search(np.array(q_emb), top_k=20 if summary_mode else top_k)
        hits = self._apply_filters(hits, filters)

        best = hits[0][0] if hits else 0.0
        top3 = [h[0] for h in hits[:3]]
        top3_avg = sum(top3) / len(top3) if top3 else 0.0
        unique_sources = len(set([h[1].source for h in hits])) if hits else 0

        citations = []
        for score, rec in hits[:4]:
            citations.append(
                {
                    "source": rec.source,
                    "page": rec.page,
                    "chunk_id": rec.chunk_id,
                    "score": round(score, 4),
                    "quote": rec.text[:320] + ("..." if len(rec.text) > 320 else ""),
                }
            )

        grounded = best >= 0.18 and top3_avg >= 0.13 and unique_sources >= 1

        if not grounded:
            return {
                "answer": "I couldn’t find enough strong support in the indexed document content to answer confidently. Try selecting a document or asking a more specific question.",
                "grounded": False,
                "confidence": 0.0,
                "citations": citations,
                "contradiction": False,
                "audit": {
                    "best_score": round(best, 4),
                    "top3_avg": round(top3_avg, 4),
                    "unique_sources": unique_sources,
                    "summary_mode": summary_mode,
                    "source_mode": False,
                    "filters": filters.__dict__,
                    "model": settings.OLLAMA_MODEL,
                    "latency_s": round(time.time() - t0, 3),
                },
            }

        context = "\n\n".join(
            [f"[{c['source']} p{c['page']}]\n{c['quote']}" for c in citations]
        )

        prompt = (
            "You are VaultQA, a document-only assistant.\n"
            "Use ONLY the provided context.\n"
            "Do not use outside knowledge.\n"
            "Answer clearly and faithfully.\n"
            "At the end, include short source references like (source p#).\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )

        import requests

        r = requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )
        r.raise_for_status()
        answer = r.json().get("response", "").strip()

        confidence = min(0.99, max(0.0, (best - 0.18) / (0.45 - 0.18)))

        return {
            "answer": answer,
            "grounded": True,
            "confidence": round(confidence, 3),
            "citations": citations,
            "contradiction": False,
            "audit": {
                "best_score": round(best, 4),
                "top3_avg": round(top3_avg, 4),
                "unique_sources": unique_sources,
                "summary_mode": summary_mode,
                "source_mode": False,
                "filters": filters.__dict__,
                "model": settings.OLLAMA_MODEL,
                "latency_s": round(time.time() - t0, 3),
            },
        }