"""
Quick test script to verify your large PDFs are being extracted correctly.

Usage:
    python test_pdf.py <path_to_pdf>
    
Example:
    python test_pdf.py data/docs/my_large_report.pdf
    python test_pdf.py /Users/me/Downloads/500page_document.pdf
"""

import sys
import time
from pathlib import Path
from backend.app.rag.pdf_extract import extract_pdf_pages
from backend.app.rag.chunking import chunk_text


def analyze_pdf(pdf_path: str):
    """Analyze a PDF and report extraction metrics."""
    
    path = Path(pdf_path)
    if not path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)
    
    if not path.suffix.lower() == ".pdf":
        print(f"❌ Error: Not a PDF file: {pdf_path}")
        sys.exit(1)
    
    file_size_mb = path.stat().st_size / (1024 * 1024)
    print("=" * 70)
    print(f"ANALYZING: {path.name}")
    print(f"Size: {file_size_mb:.2f} MB")
    print("=" * 70)
    
    # Extract pages
    print("\n🔍 Extracting pages...")
    t0 = time.time()
    try:
        pages = extract_pdf_pages(str(path))
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        sys.exit(1)
    
    extraction_time = time.time() - t0
    
    if not pages:
        print("❌ No pages extracted! The PDF might be:")
        print("   - Encrypted or password-protected")
        print("   - Empty or corrupted")
        print("   - Pure image-based (scanned) without OCR support")
        sys.exit(1)
    
    # Analyze content
    print(f"✓ Extracted {len(pages)} pages in {extraction_time:.2f}s\n")
    
    # Chunk analysis
    print("📊 Content Analysis:")
    print(f"   Total pages: {len(pages)}")
    
    total_chars = 0
    total_chunks = 0
    pages_with_content = 0
    empty_pages = 0
    
    for i, page in enumerate(pages, start=1):
        text = page["text"]
        char_count = len(text)
        total_chars += char_count
        
        if char_count > 100:
            pages_with_content += 1
        else:
            empty_pages += 1
        
        chunks = chunk_text(text)
        total_chunks += len(chunks)
        
        if i <= 3 or i > len(pages) - 3:  # Show first 3 and last 3 pages
            preview = text[:80].replace('\n', ' ') + "..."
            print(f"\n   Page {i}: {char_count:,} chars, {len(chunks)} chunks")
            print(f"   Preview: {preview}")
    
    avg_chars_per_page = total_chars // len(pages) if pages else 0
    avg_chunks_per_page = total_chunks // len(pages) if pages else 0
    chars_per_sec = total_chars / extraction_time if extraction_time > 0 else 0
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"✓ Pages extracted: {len(pages)}")
    print(f"  - Pages with content (>100 chars): {pages_with_content}")
    print(f"  - Empty/minimal pages: {empty_pages}")
    print(f"\n✓ Total text: {total_chars:,} characters")
    print(f"\n✓ Total chunks: {total_chunks} (chunk size: 900 chars, overlap: 160)")
    print(f"\n✓ Metrics:")
    print(f"  - Avg chars/page: {avg_chars_per_page:,}")
    print(f"  - Avg chunks/page: {avg_chunks_per_page}")
    print(f"  - Extraction speed: {chars_per_sec:,.0f} chars/sec")
    print(f"  - Total extraction time: {extraction_time:.2f}s")
    
    # Check if extraction was good
    print("\n" + "=" * 70)
    quality_score = pages_with_content / len(pages) * 100 if pages else 0
    
    if quality_score >= 95:
        print("✅ EXCELLENT: PDF extracted successfully!")
        print("   Ready for ingestion and summarization.")
    elif quality_score >= 75:
        print("⚠️  GOOD: Most pages extracted successfully.")
        print("   May have some scanned or image-heavy pages.")
    elif quality_score >= 50:
        print("⚠️  FAIR: About half the pages have content.")
        print("   PDF may contain many scanned images or complex layouts.")
    else:
        print("❌ POOR: Less than half the pages have extractable text.")
        print("   PDF is likely scanned or heavily image-based.")
        print("   Consider: ")
        print("   1. Installing Tesseract for OCR support")
        print("   2. Using a different PDF editor to convert/optimize")
        print("   3. Checking if PDF is encrypted")
    
    print("=" * 70)
    
    return pages, total_chunks


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    analyze_pdf(pdf_path)


if __name__ == "__main__":
    main()
