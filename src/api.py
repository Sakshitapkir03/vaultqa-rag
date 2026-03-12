from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaGenerator


DOC_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"

app = FastAPI(title="Local RAG API (Haystack + Ollama)")

store: Optional[InMemoryDocumentStore] = None

doc_embedder = SentenceTransformersDocumentEmbedder(model=DOC_EMBED_MODEL)
text_embedder = SentenceTransformersTextEmbedder(model=DOC_EMBED_MODEL)
doc_embedder.warm_up()
text_embedder.warm_up()


class IngestRequest(BaseModel):
    title: str
    content: str


class AskRequest(BaseModel):
    question: str


def ensure_store() -> InMemoryDocumentStore:
    global store
    if store is None:
        store = InMemoryDocumentStore(embedding_similarity_function="cosine")
    return store


def embed_and_store(docs: List[Document]):
    s = ensure_store()
    embedded = doc_embedder.run(documents=docs)["documents"]
    s.write_documents(embedded)


def load_txt_folder(folder: str = "data/docs"):
    p = Path(folder)
    if not p.exists():
        return
    docs: List[Document] = []
    for f in sorted(p.glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(content=text, meta={"source": f.name}))
    if docs:
        embed_and_store(docs)


@app.on_event("startup")
def startup():
    # Auto-index any docs already placed in data/docs
    load_txt_folder("data/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest):
    doc = Document(content=req.content, meta={"source": req.title})
    embed_and_store([doc])
    return {"status": "ok", "indexed": req.title}


@app.post("/ask")
def ask(req: AskRequest):
    s = ensure_store()
    retriever = InMemoryEmbeddingRetriever(document_store=s, top_k=3)
    llm = OllamaGenerator(model=LLM_MODEL)

    q_emb = text_embedder.run(text=req.question)["embedding"]
    docs = retriever.run(query_embedding=q_emb)["documents"]

    context = "\n\n".join(
        [f"[{d.meta.get('source','unknown')}]\n{d.content}" for d in docs]
    )

    prompt = (
        "You are a helpful assistant.\n"
        "Use ONLY the context below. If the answer isn't in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {req.question}\n"
        "ANSWER:"
    )

    out = llm.run(prompt=prompt)
    return {
        "answer": out["replies"][0],
        "sources": [d.meta.get("source", "unknown") for d in docs],
    }