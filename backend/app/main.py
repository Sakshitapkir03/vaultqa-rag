from __future__ import annotations

from typing import Optional
import json

import numpy as np
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import settings
from .middleware import RequestIdMiddleware
from .rag.engine import RAGEngine, AskFilters
from .rag.deep_research import DeepResearchEngine
from .rag.store import UPLOAD_DIR, load_docs_registry, ensure_dirs

app = FastAPI(title="VaultQA API", version="1.0.0")

app.add_middleware(RequestIdMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ensure_dirs()

engine = RAGEngine()
research_engine = DeepResearchEngine(engine.embedder, engine.store)


class UploadResponse(BaseModel):
    status: str
    filename: str
    sha256: str


class IndexRequest(BaseModel):
    filename: str
    collection: str = "default"
    doc_type: Optional[str] = None


class AskRequest(BaseModel):
    question: str
    collection: Optional[str] = None
    doc_type: Optional[str] = None
    source: Optional[str] = None


class StreamAskRequest(BaseModel):
    question: str
    source: Optional[str] = None


class ResearchRequest(BaseModel):
    question: str
    source: Optional[str] = None


@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error. request_id={rid}"}
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": settings.OLLAMA_MODEL,
    }


@app.get("/library")
def library():
    uploaded = sorted([p.name for p in UPLOAD_DIR.glob("*") if p.is_file()])
    reg = load_docs_registry()
    return {
        "uploaded": uploaded,
        "indexed": reg.get("docs", []),
    }


def _validate_upload(file: UploadFile, size_bytes: int):
    allowed = [e.strip().lower() for e in settings.ALLOWED_EXTS.split(",") if e.strip()]
    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(status_code=400, detail=f"Unsupported file type. Allowed: {allowed}")

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Max {settings.MAX_UPLOAD_MB}MB")


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    _validate_upload(file, len(content))

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    dest.write_bytes(content)

    import hashlib
    sha = hashlib.sha256(content).hexdigest()

    return {
        "status": "ok",
        "filename": file.filename,
        "sha256": sha,
    }


@app.post("/index")
def index(req: IndexRequest):
    return engine.ingest_file(req.filename, req.collection, req.doc_type)


@app.post("/ask")
def ask(req: AskRequest):
    filters = AskFilters(
        doc_type=req.doc_type,
        collection=req.collection,
        source=req.source,
    )
    return engine.ask(req.question, filters)


@app.post("/ask/stream")
def ask_stream(req: StreamAskRequest):
    question = req.question.strip()
    filters = AskFilters(source=req.source)
    source_mode = bool(filters.source)

    if source_mode:
        source_chunks = engine.store.get_chunks_by_source(filters.source)
        if not source_chunks:
            def empty_gen():
                yield "I couldn't find any indexed chunks for the selected document."
            return StreamingResponse(empty_gen(), media_type="text/plain")

        picked = source_chunks[:4]
        citations = []
        for rec in picked:
            citations.append({
                "source": rec.source,
                "page": rec.page,
                "chunk_id": rec.chunk_id,
                "score": 0.2,
                "quote": rec.text[:280] + ("..." if len(rec.text) > 280 else ""),
            })

        context = "\n\n".join(
            [f"[{c['source']} p{c['page']}]\n{c['quote']}" for c in citations]
        )

        prompt = (
            "You are VaultQA, a document-only assistant.\n"
            "Use ONLY the provided context.\n"
            "Do not use outside knowledge.\n"
            "Answer clearly and faithfully from the document.\n"
            "At the end, include short source references like (source p#).\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {question}\n"
            "ANSWER:"
        )
    else:
        q_emb = engine.embedder.encode([question], show_progress_bar=False)
        hits = engine.store.search(np.array(q_emb), top_k=6)

        citations = []
        for score, rec in hits[:3]:
            citations.append({
                "source": rec.source,
                "page": rec.page,
                "chunk_id": rec.chunk_id,
                "score": round(float(score), 4),
                "quote": rec.text[:280] + ("..." if len(rec.text) > 280 else ""),
            })

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

    def generate():
        with requests.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={
                "model": settings.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": True,
            },
            stream=True,
            timeout=120,
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                token = data.get("response", "")
                if token:
                    yield token

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/research")
def research(req: ResearchRequest):
    return research_engine.run(req.question, req.source or "")


@app.post("/research/stream")
def research_stream(req: ResearchRequest):
    def generate():
        for event in research_engine.stream_run(req.question, req.source or ""):
            yield json.dumps(event) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")