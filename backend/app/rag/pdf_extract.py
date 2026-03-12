from __future__ import annotations
from typing import List, Tuple
from pypdf import PdfReader

def extract_pdf_pages(path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(path)
    out = []
    for i, page in enumerate(reader.pages, start=1):
        txt = page.extract_text() or ""
        txt = txt.strip()
        if txt:
            out.append((i, txt))
    return out
