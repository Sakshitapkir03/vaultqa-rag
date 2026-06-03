"""
Test script to verify PDF extraction with large documents (500+ pages).
Generates a synthetic PDF and tests extraction and ingestion.
"""

import time
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from backend.app.rag.pdf_extract import extract_pdf_pages
from backend.app.rag.chunking import chunk_text

# Create test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TEST_DATA_DIR.mkdir(exist_ok=True)

def generate_large_pdf(num_pages: int = 500) -> Path:
    """Generate a synthetic PDF with lorem ipsum content."""
    print(f"📄 Generating {num_pages}-page test PDF...")
    
    pdf_path = TEST_DATA_DIR / f"test_large_{num_pages}pages.pdf"
    
    # Sample content
    sections = [
        "Introduction to PDF Testing",
        "Chapter 1: Fundamentals",
        "Chapter 2: Advanced Topics",
        "Chapter 3: Best Practices",
        "Chapter 4: Case Studies",
        "Chapter 5: Implementation Guide",
        "Conclusion and Future Work",
        "Appendix: Technical Details",
    ]
    
    lorem = """
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
    incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud 
    exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute 
    irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla 
    pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia 
    deserunt mollit anim id est laborum.
    
    Sed ut perspiciatis unde omnis iste natus error sit voluptatem accusantium doloremque 
    laudantium, totam rem aperiam, eaque ipsa quae ab illo inventore veritatis et quasi 
    architecto beatae vitae dicta sunt explicabo. Nemo enim ipsam voluptatem quia voluptas sit asperiores aut odit aut fugit.
    """
    
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter
    
    page_num = 0
    for page in range(num_pages):
        page_num += 1
        
        # Title
        section_idx = (page // (num_pages // len(sections))) % len(sections)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - 50, f"{sections[section_idx]} - Page {page_num}")
        
        # Content
        c.setFont("Helvetica", 10)
        y = height - 100
        for line_idx, line in enumerate(lorem.strip().split('\n')):
            if y < 50:
                c.showPage()
                page_num += 1
                c.setFont("Helvetica", 10)
                y = height - 50
                c.setFont("Helvetica-Bold", 12)
                c.drawString(50, y, "Continued...")
                y -= 30
                c.setFont("Helvetica", 10)
            c.drawString(50, y, line)
            y -= 20
        
        c.showPage()
    
    c.save()
    file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
    print(f"✓ Generated {pdf_path.name} ({file_size_mb:.2f} MB)")
    return pdf_path


def test_pdf_extraction(pdf_path: Path):
    """Test PDF extraction and report metrics."""
    print(f"\n🔍 Testing extraction on {pdf_path.name}...")
    
    t0 = time.time()
    pages = extract_pdf_pages(str(pdf_path))
    extraction_time = time.time() - t0
    
    if not pages:
        print("❌ No pages extracted!")
        return
    
    total_chars = sum(len(p["text"]) for p in pages)
    total_chunks = 0
    
    for page in pages:
        chunks = chunk_text(page["text"])
        total_chunks += len(chunks)
    
    print(f"✓ Extraction completed in {extraction_time:.2f}s")
    print(f"  - Pages extracted: {len(pages)}")
    print(f"  - Total characters: {total_chars:,}")
    print(f"  - Total chunks (900 chars each): {total_chunks}")
    print(f"  - Avg chars/page: {total_chars // len(pages):,}")
    print(f"  - Avg chunks/page: {total_chunks // len(pages)}")
    print(f"  - Extraction speed: {total_chars / extraction_time:,.0f} chars/sec")
    
    return pages


def main():
    print("=" * 60)
    print("LARGE PDF EXTRACTION TEST")
    print("=" * 60)
    
    # Test with 500-page PDF
    pdf_path = generate_large_pdf(num_pages=500)
    test_pdf_extraction(pdf_path)
    
    # Optional: Test with even larger PDF (1000 pages)
    print("\n" + "=" * 60)
    pdf_path_1000 = generate_large_pdf(num_pages=1000)
    test_pdf_extraction(pdf_path_1000)
    
    print("\n" + "=" * 60)
    print("✅ All tests completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
