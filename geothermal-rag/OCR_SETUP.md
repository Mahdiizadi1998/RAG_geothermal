# OCR Support for Scanned PDFs

## Problem: No Chunks Created from PDFs

If you see:
```
INFO:agents.preprocessing_agent:Chunking document: file.pdf
INFO:agents.preprocessing_agent:  factual_qa: 0 chunks
INFO:agents.preprocessing_agent:  technical_extraction: 0 chunks
```

Your PDF is likely **scanned** (images, not selectable text) or contains data in **tables/diagrams as images**.

---

## Solution: Enable OCR (Optical Character Recognition)

OCR extracts text from images in PDFs.

### Step 1: Install Tesseract OCR

**Windows:**
1. Download installer: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer (e.g., `tesseract-ocr-w64-setup-5.3.3.20231005.exe`)
3. **Important**: Note the installation path (default: `C:\Program Files\Tesseract-OCR`)
4. Add to PATH:
   ```cmd
   setx PATH "%PATH%;C:\Program Files\Tesseract-OCR"
   ```
5. Close and reopen terminal

**Linux (Debian/Ubuntu):**
```bash
sudo apt update
sudo apt install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Step 2: Install Python Packages

```bash
cd geothermal-rag
pip install pytesseract Pillow
```

Or reinstall all requirements:
```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
tesseract --version
```

Should output:
```
tesseract 5.x.x
```

**Python test:**
```python
python -c "import pytesseract; print('OCR available')"
```

### Step 4: Restart Application

```bash
python app.py
```

You should see:
```
INFO:agents.ingestion_agent:  ✓ OCR applied to extract text from images
```

---

## How OCR Works in This System

**Automatic Detection:**
1. System tries to extract text normally
2. If page has **< 50 characters**, OCR is triggered
3. Page is rendered as image (300 DPI)
4. Tesseract extracts text from image
5. Better result is kept

**What Gets OCR'd:**
- ✅ Scanned pages (photos of documents)
- ✅ Images containing text
- ✅ Tables rendered as images
- ✅ Diagrams with labels
- ✅ Screenshots embedded in PDFs

**What OCR Can't Handle:**
- ❌ Handwritten notes (unless very clear)
- ❌ Very low resolution images (<200 DPI)
- ❌ Heavily distorted or rotated text
- ❌ Text in non-English languages (without language pack)

---

## Performance Impact

| Document Type | No OCR | With OCR |
|--------------|--------|----------|
| Text-based PDF (20 pages) | 5 sec | 5 sec (no change) |
| Scanned PDF (20 pages) | 0 chunks ❌ | 45-60 sec ✅ |
| Mixed (10 text + 10 scanned) | 10 chunks | 60-90 sec (all 20) |

**Trade-off:**
- ✅ Extract data from scanned documents
- ✅ Get text from image-based tables
- ❌ 10-15 seconds per scanned page
- ❌ Increased CPU usage

---

## Troubleshooting

### "pytesseract.pytesseract.TesseractNotFoundError"

**Cause**: Tesseract not in PATH or not installed

**Fix:**
```cmd
# Windows: Find where Tesseract is installed
where tesseract

# If not found, add to PATH manually:
set PATH=%PATH%;C:\Program Files\Tesseract-OCR

# Or set in Python (temporary):
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
```

### "Still getting 0 chunks after OCR"

**Possible causes:**
1. **PDF is completely blank**
   - Check: Open PDF in reader, can you see any text/images?

2. **OCR quality is poor**
   - Check logs for: `WARNING:agents.ingestion_agent:Very little text extracted`
   - Try: Increase DPI in `ingestion_agent.py` (300 → 400)

3. **Text is in images but OCR failed**
   - Check: Are images clear and high resolution?
   - Try: Pre-process PDF to enhance contrast

4. **Language not English**
   - Install language pack:
     ```bash
     # German
     sudo apt install tesseract-ocr-deu
     # Dutch
     sudo apt install tesseract-ocr-nld
     ```
   - Modify code to use: `pytesseract.image_to_string(img, lang='nld')`

### "OCR text quality is poor"

**Improve OCR quality:**

1. **Increase DPI** in `agents/ingestion_agent.py`:
   ```python
   # Change from 300 to 400 or 600
   pix = page.get_pixmap(matrix=fitz.Matrix(400/72, 400/72))
   ```

2. **Pre-process images** (requires opencv):
   ```python
   # Add grayscale + threshold
   import cv2
   img_array = np.frombuffer(img_data, dtype=np.uint8)
   img_cv = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
   _, img_thresh = cv2.threshold(img_cv, 150, 255, cv2.THRESH_BINARY)
   ```

3. **Use better Tesseract model**:
   ```bash
   # Download LSTM model (more accurate, slower)
   tesseract --oem 1  # LSTM only
   ```

---

## Alternative: Manual PDF Conversion

If OCR is too slow or inaccurate:

**Option 1: Use external OCR tool first**
1. Adobe Acrobat: File → Export To → Text
2. Online: https://ocr.space or https://www.onlineocr.net
3. Save as text-based PDF
4. Upload to system

**Option 2: Extract tables separately**
1. Use Tabula: https://tabula.technology/
2. Extract tables to CSV
3. Manually copy data into text file
4. Upload text file instead

**Option 3: Request text-based reports**
- Ask data provider for searchable PDFs
- Many modern well reports are already text-based

---

## Testing OCR

Create a test script:

```python
from agents.ingestion_agent import IngestionAgent

agent = IngestionAgent()
docs = agent.process(['your_scanned_pdf.pdf'])

print(f"Pages: {docs[0]['pages']}")
print(f"Text length: {len(docs[0]['content'])}")
print(f"First 500 chars:\n{docs[0]['content'][:500]}")
```

Expected output:
```
Pages: 14
Text length: 23847
First 500 chars:
Well Report
BRI-GT-01
...
```

If text length is < 500 for a 14-page document, OCR didn't work properly.

---

## Summary

✅ **OCR enables processing of scanned PDFs**
✅ **Automatic detection** - only runs when needed
✅ **Extracts text from images and tables**
❌ **Slower** - 10-15 sec per scanned page
❌ **Requires Tesseract installation**

**Recommendation**: Install OCR support to handle all PDF types, especially Dutch geothermal reports which often contain scanned sections.
