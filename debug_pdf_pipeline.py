#!/usr/bin/env python3
"""
Debug script to test PDF upload, indexing, and summary generation pipeline.
Helps identify failures in PDF extraction, chunking, embedding, and retrieval.
"""

import os
import sys
import json
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.rag.engine import RAGEngine
from app.rag.pdf_extract import extract_pdf_pages
from app.rag.chunking import chunk_text
from app.config import settings

def test_pdf_extraction(pdf_path):
    """Test if PDF extraction works"""
    print(f"\n{'='*60}")
    print(f"Testing PDF Extraction: {pdf_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF not found at {pdf_path}")
        return False
    
    print(f"✓ PDF file exists ({os.path.getsize(pdf_path) / 1024 / 1024:.1f} MB)")
    
    try:
        pages = extract_pdf_pages(pdf_path)
        print(f"✓ Extraction successful: {len(pages)} pages extracted")
        
        empty_pages = sum(1 for p in pages if not p.get("text") or not p.get("text").strip())
        print(f"  - Pages with text: {len(pages) - empty_pages}/{len(pages)}")
        
        # Show first page
        if pages and pages[0].get("text"):
            first_text = pages[0]["text"][:200]
            print(f"  - First page (250 chars): {repr(first_text)}...")
        
        return pages
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_chunking(text, chunk_size=900, overlap=160):
    """Test if text chunking works"""
    print(f"\n{'='*60}")
    print(f"Testing Text Chunking (size={chunk_size}, overlap={overlap})")
    print(f"{'='*60}")
    
    print(f"Input text length: {len(text)} chars")
    
    try:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"✓ Chunking successful: {len(chunks)} chunks created")
        
        if chunks:
            print(f"  - Chunk sizes: min={min(len(c) for c in chunks)}, max={max(len(c) for c in chunks)}")
            print(f"  - First chunk (150 chars): {repr(chunks[0][:150])}...")
        
        return chunks
    except Exception as e:
        print(f"❌ Chunking failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_indexing(pdf_path, filename):
    """Test indexing in RAG engine"""
    print(f"\n{'='*60}")
    print(f"Testing Indexing via RAGEngine")
    print(f"{'='*60}")
    
    try:
        engine = RAGEngine()
        print(f"✓ RAGEngine initialized")
        print(f"  - Embedding model: {engine.embedder.get_sentence_embedding_dimension()} dims")
        print(f"  - Store initialized: {engine.store is not None}")
        
        # Copy PDF to upload dir if needed
        from app.rag.store import UPLOAD_DIR
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        dest = UPLOAD_DIR / filename
        
        if pdf_path != str(dest):
            import shutil
            print(f"  - Copying {pdf_path} → {dest}")
            shutil.copy(pdf_path, dest)
        
        # Ingest
        result = engine.ingest_file(filename, "default", "test")
        
        print(f"\nIndexing result:")
        print(f"  - Status: {result.get('status')}")
        if result.get('status') == 'ok':
            print(f"  ✓ Indexed {result.get('indexed_chunks', 0)} chunks")
            print(f"  - Collection: {result.get('collection')}")
        else:
            print(f"  ❌ Error: {result.get('message')}")
        
        return engine if result.get('status') == 'ok' else None
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_summary_retrieval(engine, question="Summarize this document"):
    """Test if summary retrieval works"""
    print(f"\n{'='*60}")
    print(f"Testing Summary Retrieval")
    print(f"{'='*60}")
    
    if not engine:
        print("❌ No engine provided")
        return None
    
    try:
        from app.rag.engine import AskFilters
        
        filters = AskFilters()
        result = engine.ask(question, filters)
        
        print(f"Query: {question}")
        print(f"Result Status:")
        print(f"  - Grounded: {result.get('grounded')}")
        print(f"  - Confidence: {result.get('confidence')}")
        print(f"  - Answer preview: {result.get('answer')[:150]}..." if result.get('answer') else "  - No answer")
        
        # Show audit info
        audit = result.get('audit', {})
        print(f"\nRetrieval Audit:")
        print(f"  - Best score: {audit.get('best_score')}")
        print(f"  - Top 3 avg: {audit.get('top3_avg')}")
        print(f"  - Unique sources: {audit.get('unique_sources')}")
        print(f"  - Summary mode: {audit.get('summary_mode')}")
        
        # Show citations
        citations = result.get('citations', [])
        print(f"\nCitations ({len(citations)}):")
        for c in citations[:3]:
            print(f"  - {c.get('source')} p{c.get('page')} (score: {c.get('score')})")
            print(f"    {repr(c.get('quote', '')[:100])}...")
        
        return result
    except Exception as e:
        print(f"❌ Summary retrieval failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    print("\n" + "="*60)
    print("RAG PIPELINE DEBUG SCRIPT")
    print("="*60)
    
    # Find test PDF
    from pathlib import Path
    test_dir = Path(__file__).parent / "test_data"
    pdfs = list(test_dir.glob("*.pdf"))
    
    if not pdfs:
        print("\n❌ No PDF files found in test_data/")
        print("   Please add a test PDF to test_data/")
        return
    
    pdf_path = str(pdfs[0])  # Use first PDF found
    pdf_name = Path(pdf_path).name
    
    print(f"\nUsing test PDF: {pdf_name}")
    
    # Step 1: Test extraction
    pages = test_pdf_extraction(pdf_path)
    if not pages:
        print("\n❌ Extraction failed, stopping")
        return
    
    # Step 2: Test chunking on first page
    if pages and pages[0].get("text"):
        chunks = test_chunking(pages[0]["text"])
        if not chunks:
            print("\n❌ Chunking failed, stopping")
            return
    
    # Step 3: Test indexing
    engine = test_indexing(pdf_path, pdf_name)
    if not engine:
        print("\n❌ Indexing failed, stopping")
        return
    
    # Step 4: Test summary retrieval
    test_summary_retrieval(engine, "Summarize this document")
    
    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
