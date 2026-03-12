from __future__ import annotations
from typing import List, Dict, Any

def chunk_text(text: str, chunk_size: int = 1100, overlap: int = 220) -> List[str]:
    # normalize whitespace
    text = " ".join(text.split())
    chunks = []
    i = 0
    while i < len(text):
        chunks.append(text[i:i+chunk_size])
        i += max(1, chunk_size - overlap)
    return chunks

def guess_doc_type(filename: str) -> str:
    fn = filename.lower()
    if "policy" in fn: return "policy"
    if "manual" in fn: return "manual"
    if "report" in fn: return "report"
    if "paper" in fn or "research" in fn: return "paper"
    if "note" in fn or "lecture" in fn: return "notes"
    return "general"
