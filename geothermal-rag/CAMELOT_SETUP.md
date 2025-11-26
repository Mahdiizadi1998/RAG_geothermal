# Camelot-py Setup Guide

## Overview

The system now uses **camelot-py** for high-accuracy table extraction from PDFs. Camelot provides:
- ✅ Accuracy scoring (0-100%) for each table
- ✅ Two detection modes: Lattice (line-based) and Stream (whitespace-based)
- ✅ Better handling of complex table layouts
- ✅ Automatic rejection of headers/footers via low accuracy scores
- ✅ Visual debugging tools

## System Dependencies

Camelot requires **Ghostscript** to be installed on your system.

### Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y ghostscript python3-tk
```

### macOS

```bash
brew install ghostscript tcl-tk
```

### Windows

1. Download Ghostscript installer from: https://ghostscript.com/releases/gsdnld.html
2. Install Ghostscript (use default installation path)
3. Add to PATH:
   - Right-click "This PC" → Properties → Advanced System Settings
   - Environment Variables → System Variables → Path → Edit
   - Add: `C:\Program Files\gs\gs10.02.0\bin` (adjust version number)
4. Restart terminal/IDE

### Docker

If using Docker, update your Dockerfile:

```dockerfile
FROM python:3.9

# Install system dependencies for camelot
RUN apt-get update && apt-get install -y \
    ghostscript \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
```

## Python Dependencies

Install Python package:

```bash
pip install camelot-py[cv]
```

This installs:
- `camelot-py` - Main library
- `opencv-python` - For image processing
- `pandas` - For DataFrame support

## Verification

Test that camelot is working:

```python
import camelot

# Should print version without errors
print(f"Camelot version: {camelot.__version__}")

# Test table extraction
tables = camelot.read_pdf('test.pdf', pages='1')
print(f"Found {len(tables)} tables")
```

## Configuration

### Accuracy Thresholds

In `ingestion_agent.py`, you can adjust quality thresholds:

```python
# Lattice mode (tables with borders)
if table.accuracy >= 75:  # Default: 75%
    keep_table()

# Stream mode (borderless tables)
if table.accuracy >= 70:  # Default: 70% (slightly lower)
    keep_table()
```

**Guidelines:**
- `accuracy >= 90`: Excellent quality, high confidence
- `accuracy >= 75`: Good quality, reliable extraction
- `accuracy >= 60`: Fair quality, may need validation
- `accuracy < 60`: Poor quality, likely false positive (header/footer)

### Detection Modes

**Lattice Mode** (for tables with visible borders):
```python
tables = camelot.read_pdf(
    pdf_path,
    flavor='lattice',
    line_scale=40  # Sensitivity to line detection (default: 40)
)
```

**Stream Mode** (for borderless tables):
```python
tables = camelot.read_pdf(
    pdf_path,
    flavor='stream',
    edge_tol=50,  # Column edge tolerance (default: 50)
    row_tol=2     # Row detection tolerance (default: 2)
)
```

## Troubleshooting

### "Ghostscript not found"

**Error:**
```
GhostscriptNotFound: Please make sure that Ghostscript is installed
```

**Solution:**
1. Install Ghostscript (see above)
2. Verify installation: `gs --version`
3. Ensure it's in PATH

### Low Accuracy Scores

If tables are being rejected due to low accuracy:

1. **Check table structure in PDF**
   - Are lines clearly visible?
   - Is text properly aligned?
   - Are there merged cells?

2. **Try different mode**
   - Lattice for tables with borders
   - Stream for borderless tables

3. **Adjust thresholds**
   ```python
   # Lower threshold if missing real tables
   if table.accuracy >= 65:  # Instead of 75
       keep_table()
   ```

4. **Use visual debugging**
   ```python
   table.plot('lattice').save('debug.png')
   # Opens image showing detected lines/regions
   ```

### Slow Extraction

Camelot is slower than pdfplumber (1-2s vs 0.2s per page).

**Optimization strategies:**
1. Process only specific pages: `pages='15-20'`
2. Use one mode only (lattice OR stream, not both)
3. Parallel processing for multiple PDFs

**Trade-off:** Slower extraction, but much higher accuracy (90% vs 70%)

### Memory Issues

Camelot uses ~150MB RAM per PDF (due to image conversion).

**Solution:**
- Process PDFs one at a time
- Clear memory between documents
- Increase Docker container memory limit

## Comparison: pdfplumber vs camelot

| Aspect | pdfplumber | camelot |
|--------|-----------|---------|
| **Speed** | ⭐⭐⭐ Fast (200ms/page) | ⭐⭐ Medium (1-2s/page) |
| **Accuracy** | ⭐⭐⭐ Good (70-80%) | ⭐⭐⭐⭐ Excellent (85-95%) |
| **Dependencies** | ✅ Python only | ⚠️ Requires Ghostscript |
| **Quality Scoring** | ❌ No | ✅ Yes (0-100%) |
| **Visual Debug** | ❌ No | ✅ Yes (plot tools) |
| **2-col Tables** | ⚠️ Hard to detect | ✅ Detects reliably |
| **Headers/Footers** | ⚠️ Need manual filter | ✅ Auto-rejected (low score) |
| **Complex Layouts** | ⚠️ Struggles | ✅ Handles well |

## Migration Notes

### Changes from pdfplumber

1. **No more strict column filtering**
   - Old: `if len(headers) >= 3` (missed 2-col tables)
   - New: Accuracy-based filtering (works for any table)

2. **Automatic quality control**
   - Old: Manual validation rules
   - New: Built-in accuracy scoring

3. **Better multi-table handling**
   - Old: Tables on same page sometimes merged
   - New: Separate detection with overlap checking

4. **Header/footer rejection**
   - Old: Manual region filtering (top 10%, bottom 10%)
   - New: Automatic via low accuracy scores

### What Stays the Same

- Table storage in database (unchanged)
- Well name detection (unchanged)
- Table classification with LLM (unchanged)
- Formatted output structure (unchanged)

## Support

For issues specific to camelot:
- GitHub: https://github.com/camelot-dev/camelot
- Documentation: https://camelot-py.readthedocs.io/
- Known issues: Check GitHub Issues tab

For your system integration issues:
- Check logs for accuracy scores
- Use visual debugging: `table.plot('lattice')`
- Adjust thresholds if needed
