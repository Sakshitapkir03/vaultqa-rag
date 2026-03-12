from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import faiss

INDEX_DIR = Path("data/index")
UPLOAD_DIR = Path("data/uploads")
META_JSONL = INDEX_DIR / "chunks.jsonl"
DOCS_JSON = INDEX_DIR / "docs.json"
FAISS_INDEX = INDEX_DIR / "faiss.index"

def ensure_dirs():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if not DOCS_JSON.exists():
        DOCS_JSON.write_text(json.dumps({"docs": []}, indent=2), encoding="utf-8")
    if not META_JSONL.exists():
        META_JSONL.write_text("", encoding="utf-8")

def normalize(v: np.ndarray) -> np.ndarray:
    # cosine similarity via inner product on unit vectors
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms

@dataclass
class ChunkRecord:
    chunk_id: str
    source: str
    page: int
    doc_type: str
    collection: str
    text: str

    def to_json(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "page": self.page,
            "doc_type": self.doc_type,
            "collection": self.collection,
            "text": self.text,
        }

class VectorStore:
    """
    Persistent FAISS index + JSONL chunk metadata.
    """
    def __init__(self, dim: int):
        ensure_dirs()
        self.dim = dim
        self.index = self._load_or_create_index(dim)
        self._chunk_count = self.index.ntotal
        self._id_map = self._load_all_chunk_meta()  # list index -> ChunkRecord

    def _load_or_create_index(self, dim: int):
        if FAISS_INDEX.exists():
            return faiss.read_index(str(FAISS_INDEX))
        # Flat inner product (cosine after normalization)
        idx = faiss.IndexFlatIP(dim)
        faiss.write_index(idx, str(FAISS_INDEX))
        return idx

    def _load_all_chunk_meta(self) -> List[ChunkRecord]:
        records: List[ChunkRecord] = []
        if META_JSONL.exists():
            with META_JSONL.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    records.append(ChunkRecord(**obj))
        return records

    def persist(self):
        faiss.write_index(self.index, str(FAISS_INDEX))

    def add_embeddings(self, embeddings: np.ndarray, records: List[ChunkRecord]):
        assert embeddings.shape[0] == len(records)
        emb = normalize(embeddings.astype("float32"))
        self.index.add(emb)
        # append metadata
        with META_JSONL.open("a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r.to_json(), ensure_ascii=False) + "\n")
        self._id_map.extend(records)
        self._chunk_count = self.index.ntotal
        self.persist()

    def search(self, query_emb: np.ndarray, top_k: int = 6) -> List[Tuple[float, ChunkRecord]]:
        q = normalize(query_emb.astype("float32"))
        scores, idxs = self.index.search(q, top_k)
        results = []
        for score, ix in zip(scores[0].tolist(), idxs[0].tolist()):
            if ix == -1:
                continue
            results.append((float(score), self._id_map[ix]))
        return results

    def get_chunks_by_source(self, source: str):
        return [rec for rec in self._id_map if rec.source == source]

def load_docs_registry() -> Dict[str, Any]:
    ensure_dirs()
    return json.loads(DOCS_JSON.read_text(encoding="utf-8"))

def save_docs_registry(obj: Dict[str, Any]):
    DOCS_JSON.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def register_doc(doc_meta: Dict[str, Any]):
    reg = load_docs_registry()
    docs = reg.get("docs", [])
    # replace if exists
    docs = [d for d in docs if d.get("filename") != doc_meta.get("filename")]
    docs.append(doc_meta)
    reg["docs"] = sorted(docs, key=lambda x: x.get("filename", ""))
    save_docs_registry(reg)