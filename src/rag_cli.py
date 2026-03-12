from pathlib import Path
from typing import List

from haystack.dataclasses import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaGenerator


DOC_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama3.1:8b"  # change if you pulled a different ollama model


def load_txt_docs(folder: str) -> List[Document]:
    p = Path(folder)
    if not p.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    docs: List[Document] = []
    for f in sorted(p.glob("*.txt")):
        text = f.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append(Document(content=text, meta={"source": f.name}))
    return docs


def build_store(docs: List[Document]) -> InMemoryDocumentStore:
    store = InMemoryDocumentStore(embedding_similarity_function="cosine")

    doc_embedder = SentenceTransformersDocumentEmbedder(model=DOC_EMBED_MODEL)
    doc_embedder.warm_up()

    embedded_docs = doc_embedder.run(documents=docs)["documents"]
    store.write_documents(embedded_docs)
    return store


def answer(store: InMemoryDocumentStore, question: str) -> str:
    text_embedder = SentenceTransformersTextEmbedder(model=DOC_EMBED_MODEL)
    text_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(document_store=store, top_k=3)
    llm = OllamaGenerator(model=LLM_MODEL)

    q_emb = text_embedder.run(text=question)["embedding"]
    retrieved = retriever.run(query_embedding=q_emb)["documents"]

    context = "\n\n".join(
        [f"[Source: {d.meta.get('source','unknown')}]\n{d.content}" for d in retrieved]
    )

    prompt = (
        "You are a helpful assistant.\n"
        "Use ONLY the context below. If the answer isn't in the context, say you don't know.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {question}\n"
        "ANSWER:"
    )

    out = llm.run(prompt=prompt)
    answer_text = out["replies"][0]

    print("\n--- Retrieved sources ---")
    for i, d in enumerate(retrieved, 1):
        print(f"{i}) {d.meta.get('source','unknown')}")

    return answer_text


def main():
    docs = load_txt_docs("data/docs")
    store = build_store(docs)

    print("\n✅ Local RAG CLI ready.")
    print("Type a question. Type 'exit' to quit.")

    while True:
        q = input("\nQ> ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        a = answer(store, q)
        print("\nA>", a)


if __name__ == "__main__":
    main()