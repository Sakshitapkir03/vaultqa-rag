from __future__ import annotations

import re
from typing import List, Dict, Any


def infer_page_metadata(page_text: str) -> Dict[str, Any]:
    text = (page_text or "").strip()
    lower = text.lower()

    section_type = "body"
    section_title = None
    year = None
    heading_path = None

    if not text:
        section_type = "unknown"
        return {
            "section_title": section_title,
            "section_type": section_type,
            "year": year,
            "heading_path": heading_path,
        }

    if "table of contents" in lower or lower.startswith("contents") or "\ncontents" in lower:
        section_type = "toc"
    elif lower.startswith("appendix") or "\nappendix" in lower:
        section_type = "appendix"
    elif lower.startswith("index") or "\nindex" in lower:
        section_type = "index"
    elif "references" in lower or "bibliography" in lower:
        section_type = "references"
    elif "all rights reserved" in lower or "published by" in lower or "copyright" in lower:
        section_type = "front_matter"

    year_match = re.search(r"\b(?:19|20)\d{2}\b", text)
    if year_match:
        year = year_match.group(0)

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        first_line = lines[0]
        if len(first_line) <= 120:
            section_title = first_line
            heading_path = first_line

    return {
        "section_title": section_title,
        "section_type": section_type,
        "year": year,
        "heading_path": heading_path,
    }


def _langchain_pypdf_extract(path: str) -> List[str]:
    """Extract page texts using LangChain's PyPDFLoader (backed by pypdf)."""
    try:
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(path)
        docs = loader.load()
        if not docs:
            return []
        max_page = max(doc.metadata.get("page", 0) for doc in docs)
        page_texts: List[str] = [""] * (max_page + 1)
        for doc in docs:
            page_num = doc.metadata.get("page", 0)
            page_texts[page_num] = doc.page_content.strip()
        return page_texts
    except Exception:
        return []


def _langchain_pymupdf_extract(path: str) -> List[str]:
    """Fallback extraction using LangChain's PyMuPDFLoader (backed by pymupdf/fitz)."""
    try:
        from langchain_community.document_loaders import PyMuPDFLoader
    except ImportError:
        return []
    try:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        if not docs:
            return []
        max_page = max(doc.metadata.get("page", 0) for doc in docs)
        page_texts: List[str] = [""] * (max_page + 1)
        for doc in docs:
            page_num = doc.metadata.get("page", 0)
            page_texts[page_num] = doc.page_content.strip()
        return page_texts
    except Exception:
        return []


def extract_pdf_pages(path: str) -> List[Dict[str, Any]]:
    """Extract pages from a PDF using LangChain document loaders.

    Uses PyPDFLoader as the primary loader and falls back to PyMuPDFLoader
    when more than 25% of pages are empty (common with scanned PDFs).
    """
    page_texts = _langchain_pypdf_extract(path)

    total_count = len(page_texts)
    empty_count = sum(1 for t in page_texts if not t)

    if total_count > 0 and empty_count > max(2, total_count // 4):
        fitz_texts = _langchain_pymupdf_extract(path)
        if fitz_texts and len(fitz_texts) == total_count:
            page_texts = fitz_texts

    out: List[Dict[str, Any]] = []
    for i, txt in enumerate(page_texts, start=1):
        if txt:
            meta = infer_page_metadata(txt)
            out.append(
                {
                    "page": i,
                    "text": txt,
                    "section_title": meta["section_title"],
                    "section_type": meta["section_type"],
                    "year": meta["year"],
                    "heading_path": meta["heading_path"],
                }
            )

    return out
