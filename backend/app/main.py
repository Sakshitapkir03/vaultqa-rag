from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load backend environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

from typing import Optional
from pathlib import Path
from .rag.store import META_JSONL, DOCS_JSON, FAISS_INDEX, INDEX_DIR
import json

import numpy as np
import requests
from fastapi import Depends
from .auth import get_current_user_id
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from .config import settings
from .middleware import RequestIdMiddleware
from .rag.engine import RAGEngine, AskFilters
from .rag.deep_research import DeepResearchEngine
from .rag.store import UPLOAD_DIR, load_docs_registry, ensure_dirs
from .storage.db import init_db
from .storage.chat_store import (
    create_conversation,
    list_conversations,
    get_conversation,
    delete_conversation,
    add_message,
    list_messages,
)


app = FastAPI(title="VaultQA API", version="1.0.0")

ensure_dirs()
init_db()

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


# -------------------------
# STARTUP CHECKS
# -------------------------

def remove_document_from_registry(filename: str):
    reg = load_docs_registry()
    docs = reg.get("docs", [])
    docs = [d for d in docs if d.get("filename") != filename]
    reg["docs"] = docs
    DOCS_JSON.write_text(json.dumps(reg, indent=2), encoding="utf-8")

@app.on_event("startup")
def verify_supabase_config():
    """
    Prevents runtime failures by validating Supabase config on startup.
    """
    supabase_url = os.getenv("SUPABASE_URL", "").strip()

    if not supabase_url:
        raise RuntimeError("SUPABASE_URL missing in backend/.env")

    jwks_url = f"{supabase_url}/auth/v1/.well-known/jwks.json"

    try:
        r = requests.get(jwks_url, timeout=10)
        r.raise_for_status()
        print(f"INFO: Supabase connection verified at {jwks_url}")
    except Exception as e:
        print(f"WARNING: Could not reach Supabase JWKS endpoint: {jwks_url}\n{e}")
        print("WARNING: Auth endpoints will not work. Core RAG features (upload/index/ask/research) are unaffected.")


# -------------------------
# MODELS
# -------------------------

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


class ConversationCreateRequest(BaseModel):
    title: str
    mode: str
    source: Optional[str] = None


class MessageCreateRequest(BaseModel):
    role: str
    content: str
    citations: Optional[list] = None
    research: Optional[dict] = None


# -------------------------
# GLOBAL ERROR HANDLER
# -------------------------

@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    rid = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error. request_id={rid}"}
    )


# -------------------------
# HEALTH
# -------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": settings.OLLAMA_MODEL,
    }


# -------------------------
# LIBRARY
# -------------------------

@app.get("/library")
def library():
    uploaded = sorted([p.name for p in UPLOAD_DIR.glob("*") if p.is_file()])
    reg = load_docs_registry()

    return {
        "uploaded": uploaded,
        "indexed": reg.get("docs", []),
    }


# -------------------------
# FILE UPLOAD
# -------------------------

def _validate_upload(file: UploadFile, size_bytes: int):
    allowed = [e.strip().lower() for e in settings.ALLOWED_EXTS.split(",") if e.strip()]

    if not any(file.filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {allowed}"
        )

    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.MAX_UPLOAD_MB}MB"
        )


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


# -------------------------
# INDEX
# -------------------------

@app.post("/index")
def index(req: IndexRequest):
    return engine.ingest_file(req.filename, req.collection, req.doc_type)


# -------------------------
# ASK
# -------------------------

@app.post("/ask")
def ask(req: AskRequest):
    filters = AskFilters(
        doc_type=req.doc_type,
        collection=req.collection,
        source=req.source,
    )

    return engine.ask(req.question, filters)


# -------------------------
# ASK STREAM
# -------------------------

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

        # Use hybrid retrieval so the streamed answer uses the most relevant chunks
        try:
            candidate_hits = engine._search_candidate_hits(question, source_chunks, False)
            picked_hits = candidate_hits[:6]
        except Exception:
            picked_hits = []

        if not picked_hits:
            context = "\n\n".join(
                [f"[{rec.source} p{rec.page}]\n{rec.text}" for rec in source_chunks[:4]]
            )
        else:
            context = "\n\n".join(
                [f"[{rec.source} p{rec.page}]\n{rec.text}" for _, rec in picked_hits]
            )

    else:

        q_emb = engine.embedder.encode([question], show_progress_bar=False)
        hits = engine.store.search(np.array(q_emb), top_k=6)

        context = "\n\n".join(
            [f"[{rec.source} p{rec.page}]\n{rec.text}" for _, rec in hits[:3]]
        )

    prompt = (
        "You are VaultQA, a document-only assistant.\n"
        "Use ONLY the provided context.\n"
        "Do not use outside knowledge.\n"
        "Answer clearly and faithfully.\n\n"
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
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 8192,
                },
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


# -------------------------
# RESEARCH
# -------------------------

@app.post("/research")
def research(req: ResearchRequest):
    return research_engine.run(req.question, req.source or "")


@app.post("/research/stream")
def research_stream(req: ResearchRequest):

    def generate():
        for event in research_engine.stream_run(req.question, req.source or ""):
            yield json.dumps(event) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


# -------------------------
# CONVERSATIONS
# -------------------------

@app.get("/conversations")
def conversations(user_id: str = Depends(get_current_user_id)):
    return {"items": list_conversations(user_id)}


@app.post("/conversations")
def create_conversation_api(
    req: ConversationCreateRequest,
    user_id: str = Depends(get_current_user_id),
):
    return create_conversation(user_id, req.title, req.mode, req.source)


@app.get("/conversations/{conversation_id}")
def get_conversation_api(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
):
    conv = get_conversation(conversation_id, user_id)

    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    return {
        "conversation": conv,
        "messages": list_messages(conversation_id, user_id),
    }


@app.delete("/conversations/{conversation_id}")
def delete_conversation_api(
    conversation_id: str,
    user_id: str = Depends(get_current_user_id),
):
    delete_conversation(conversation_id, user_id)
    return {"status": "ok"}


@app.post("/conversations/{conversation_id}/messages")
def add_message_api(
    conversation_id: str,
    req: MessageCreateRequest,
    user_id: str = Depends(get_current_user_id),
):
    return add_message(
        user_id,
        conversation_id,
        req.role,
        req.content,
        req.citations,
        req.research,
    )

@app.delete("/documents/{filename}")
def delete_document_api(filename: str):
    file_path = UPLOAD_DIR / filename

    if file_path.exists():
        file_path.unlink()

    remove_document_from_registry(filename)

    return {"status": "ok", "filename": filename}