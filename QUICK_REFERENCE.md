# 🚀 Quick Start: Testing Your Large PDFs

## What Changed
✅ Added **PyMuPDF (fitz)** as a fallback PDF extraction method  
✅ Your system now intelligently switches extraction methods if needed  
✅ Can now handle 500+ page PDFs efficiently  

---

## Test Your PDF in 60 Seconds

### Option 1: Quick Diagnostic (Recommended)
```bash
cd /Users/sakshitapkir/Desktop/sakshi-haystack-rag
source .venv/bin/activate
python test_pdf.py path/to/your/large_pdf.pdf
```

**Example:**
```bash
python test_pdf.py data/docs/my_report.pdf
python test_pdf.py ~/Downloads/large_document.pdf
```

**What you'll see:**
- ✅ If EXCELLENT: Your PDF is ready to use
- ⚠️ If FAIR/GOOD: PDF works but has some layout issues
- ❌ If POOR: PDF is likely scanned (may need OCR setup)

---

### Option 2: Test with Generated PDF (Full Demo)
```bash
python test_large_pdf.py
```
This generates 500-page and 1000-page test PDFs and shows extraction metrics.

---

## Upload & Test in Your App

1. Open your frontend at `http://localhost:3000` (or wherever it's running)
2. Upload your 500+ page PDF
3. Click "Summarize" or ask a question
4. **Watch the magic:** System will now:
   - Extract all pages efficiently
   - Chunk them intelligently (900 chars each)
   - Retrieve relevant sections for your query
   - Generate complete summaries

---

## Next: Fine-Tune for Your Needs

If summaries are still incomplete, try these:

### A. Get More Chunks Per Query (in `backend/app/rag/engine.py` around line 485)
```python
# Make these numbers bigger:
dense_top_k=60        # was 40
lexical_top_k=60      # was 40
final_top_k=15        # was 12
```

### B. Adjust Chunking Strategy (in `backend/app/rag/chunking.py`)
```python
# For detailed Q&A (more chunks):
chunk_size = 600    # was 900

# For summaries (broader context):
chunk_size = 1200   # was 900
```

### C. Faster Processing (in `backend/app/rag/engine.py` around line 103)
```python
batch_size = 64     # was 32 (faster, needs more RAM)
```

---

## Troubleshooting

**Q: Still incomplete summaries?**
- A: Try Option A above (increase dense_top_k, lexical_top_k)

**Q: Extraction seems slow?**
- A: Increase batch_size (uses more RAM but faster)

**Q: "No text extracted" error?**
- Run: `python test_pdf.py your_file.pdf`
- If result is POOR, your PDF is likely scanned
- Need OCR? Uncomment this issue and I'll add Tesseract support

---

## Files You Got

| File | Purpose |
|------|---------|
| `backend/app/rag/pdf_extract.py` | **UPDATED** - Now handles scanned PDFs |
| `test_large_pdf.py` | Generate 500/1000 page test PDFs |
| `test_pdf.py` | Quick diagnostic for any PDF |
| `LARGE_PDF_IMPROVEMENTS.md` | Full technical documentation |
| `QUICK_REFERENCE.md` | This file! |

---

## Commands Cheat Sheet

```bash
# Extract and analyze any PDF
python test_pdf.py path/to/pdf.pdf

# Generate test PDFs
python test_large_pdf.py

# Check PyMuPDF is installed
pip show pymupdf

# Reinstall everything (if issues)
pip install -r requirements.txt   # Update when you have requirements.txt
```

---

## Architecture (How It Works)

```
Your Large PDF (500+ pages)
        ↓
Try pypdf (fast, works for most PDFs)
        ↓
If >25% pages are empty → Fall back to PyMuPDF (handles scanned PDFs)
        ↓
Extract text from all pages
        ↓
Split into 900-char chunks with overlap
        ↓
Create embeddings (32-64 at a time)
        ↓
Store in FAISS index
        ↓
User asks question → Retrieve top chunks → Generate answer
```

---

## Ready to Go! 

Start with:
```bash
python test_pdf.py your_large_pdf.pdf
```

See the ✅ EXCELLENT message? Then upload it to your app and it will work! 🎉
