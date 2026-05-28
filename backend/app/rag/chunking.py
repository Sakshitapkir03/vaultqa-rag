from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def guess_doc_type(filename: str) -> str:
    lower = filename.lower()
    if "resume" in lower or "cv" in lower:
        return "resume"
    if "policy" in lower:
        return "policy"
    if "manual" in lower:
        return "manual"
    if "report" in lower:
        return "report"
    if "paper" in lower or "research" in lower:
        return "paper"
    if "notes" in lower:
        return "notes"
    return "general"


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)
