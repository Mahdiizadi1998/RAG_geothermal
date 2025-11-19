# Quick Start Guide

## System Requirements

- **Python**: 3.9 or higher
- **RAM**: 16GB recommended (8GB minimum)
- **Disk Space**: 10GB (for models and vector DB)
- **OS**: Linux, macOS, or Windows with WSL

## Installation (5 minutes)

### 1. Install Ollama

Visit [https://ollama.ai/](https://ollama.ai/) and install for your OS.

Start Ollama:
```bash
ollama serve
```

### 2. Install Dependencies

```bash
# From geothermal-rag directory
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 3. Pull Ollama Models

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

### 4. Test Installation

```bash
python test_system.py
```

Expected output:
```
Testing imports...
✓ IngestionAgent
✓ PreprocessingAgent
...
All tests completed successfully! ✓
```

## Running the System

```bash
python app.py
```

Open browser to: `http://localhost:7860`

## First Use Workflow

### Step 1: Upload Documents

1. Go to "Document Upload" tab
2. Click "Upload PDF Reports"
3. Select one or more completion reports
4. Click "Index Documents"
5. Wait for indexing (typically 20-40 seconds)

### Step 2: Ask Questions

1. Go to "Query Interface" tab
2. Select query mode:
   - **Q&A**: Specific questions
   - **Summary**: Document overview
   - **Extract & Analyze**: Parameter extraction + nodal analysis
3. Enter your question
4. Click "Submit Query"

## Example Queries

### Q&A Mode
```
What is the total depth of well ADK-GT-01?
What casing sizes were used?
What is the fluid density?
```

### Summary Mode
```
Summarize the completion report
Give me an overview of the well design
```

### Extract & Analyze Mode
```
Extract trajectory and analyze production for ADK-GT-01
Get well geometry and calculate flow capacity
```

## Test Data

Download sample geothermal well reports from:
- **NLOG Portal**: https://www.nlog.nl/
- Search: "geothermal"
- Filter: "End of Well Reports" or "Completion Reports"

Recommended test wells:
- ADK-GT-01 (Aardwarmte Den Haag)
- RNAU-GT-02 (Ridderkerk Naaldwijk)
- LEIC-GT-01 (Leidschenveen)

## Common Issues

### "Ollama connection refused"
**Solution**: Start Ollama server
```bash
ollama serve
```

### "Collection not found"
**Solution**: Index documents first in Upload tab

### Low extraction confidence
**Causes**:
- Scanned PDF (poor OCR quality)
- Non-standard table format
- Missing data in document

**Solutions**:
- Try a different PDF
- Increase chunk_size in config.yaml
- Check source document manually

### Slow performance
**Solutions**:
- Use llama3 instead of llama3.1 (faster, slightly less accurate)
- Reduce top_k in config.yaml
- Close other applications to free RAM

## Configuration

Edit `config/config.yaml` to customize:

```yaml
# Use different model
ollama:
  model_qa: llama3.1  # More accurate but slower

# Adjust chunk sizes
chunking:
  technical_extraction:
    chunk_size: 3000  # Increase for very large tables

# Adjust retrieval
retrieval:
  top_k_extraction: 30  # Retrieve more chunks
```

## Performance Tips

1. **First query is slow**: Models need to load (30-60s)
2. **Subsequent queries are fast**: Models cached in memory (2-5s)
3. **Use Extract mode for wells already indexed**: Results are cached
4. **Close browser tabs**: Saves RAM for models

## Getting Help

1. Check `README.md` for detailed documentation
2. Review `context.txt` for implementation details
3. Run `python test_system.py` to verify setup
4. Check logs in terminal for error messages

## Next Steps

After successful first run:

1. **Try different PDFs**: Test robustness with varied formats
2. **Adjust configuration**: Tune for your specific documents
3. **Explore validation**: Review extraction quality in debug output
4. **Experiment with nodal analysis**: Compare with manual calculations

---

**Ready to extract!** 🚀

For questions, refer to the comprehensive documentation in `README.md` and `context.txt`.
