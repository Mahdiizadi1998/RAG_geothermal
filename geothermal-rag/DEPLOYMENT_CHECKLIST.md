# Deployment Checklist

Use this checklist to ensure the system is properly set up and ready for use.

## ✅ Pre-Deployment Verification

### System Requirements
- [ ] Python 3.9 or higher installed (`python3 --version`)
- [ ] Pip package manager available (`pip --version`)
- [ ] 16GB RAM available (8GB minimum)
- [ ] 10GB free disk space
- [ ] Internet connection (for initial setup only)

### Ollama Setup
- [ ] Ollama installed ([https://ollama.ai/](https://ollama.ai/))
- [ ] Ollama service running (`curl http://localhost:11434/api/tags`)
- [ ] llama3 model pulled (`ollama list` shows llama3)
- [ ] nomic-embed-text model pulled (`ollama list` shows nomic-embed-text)

## 📦 Installation Steps

### Automated Setup (Recommended)
```bash
cd /workspaces/RAG_geothermal/geothermal-rag
./setup.sh
```

- [ ] Setup script completed without errors
- [ ] All Python dependencies installed
- [ ] spaCy model downloaded
- [ ] Ollama models pulled
- [ ] Test suite passed

### Manual Setup (Alternative)
```bash
# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Pull Ollama models
ollama pull llama3
ollama pull nomic-embed-text

# Run tests
python test_system.py
```

- [ ] Each step completed successfully
- [ ] No import errors
- [ ] Test output shows all green checkmarks

## 🧪 Post-Installation Testing

### Component Tests
Run the test suite:
```bash
python test_system.py
```

Expected output:
- [ ] ✓ All imports successful
- [ ] ✓ Pattern library working (trajectory & casing extraction)
- [ ] ✓ Unit conversion working (fractional inches, validations)
- [ ] ✓ Nodal analysis working (pressure calculations)

### Application Launch
Start the application:
```bash
python app.py
```

- [ ] Server starts without errors
- [ ] Gradio interface accessible at http://localhost:7860
- [ ] All three tabs visible (Document Upload, Query Interface, About)
- [ ] No Python errors in terminal

## 📄 First Document Test

### Upload Test
1. Obtain a test PDF (geothermal well report from NLOG)
2. Go to "Document Upload" tab
3. Upload PDF file

Verify:
- [ ] File upload successful
- [ ] "Index Documents" button clickable
- [ ] Click "Index Documents"
- [ ] Indexing completes (20-40 seconds)
- [ ] Status shows:
  - [ ] Documents processed count
  - [ ] Page count
  - [ ] Well names detected (if present)
  - [ ] Chunk statistics
  - [ ] "✓ Ready for queries!" message

### Query Test (Q&A Mode)
1. Go to "Query Interface" tab
2. Select "Q&A" mode
3. Enter: "What is the total depth?"
4. Click "Submit Query"

Verify:
- [ ] Response appears within 10 seconds
- [ ] Response cites source file
- [ ] Response mentions page numbers
- [ ] Debug info shows retrieved chunks
- [ ] No errors in terminal

### Query Test (Summary Mode)
1. Select "Summary" mode
2. Enter: "Summarize the document"
3. Click "Submit Query"

Verify:
- [ ] Summary generated
- [ ] Multiple sections covered
- [ ] Reasonable length (not too short)
- [ ] No hallucinated information

### Query Test (Extract & Analyze Mode)
1. Select "Extract & Analyze" mode
2. Enter: "Extract trajectory and analyze" (include well name if known)
3. Click "Submit Query"

Verify:
- [ ] Extraction results shown
- [ ] Confidence score displayed
- [ ] Trajectory data present (if in document)
- [ ] Validation results shown
- [ ] Nodal analysis section present (if data valid)
- [ ] No critical errors

## 🔧 Configuration Verification

### Config File Check
Open `config/config.yaml` and verify:
- [ ] Ollama host: http://localhost:11434
- [ ] Models specified: llama3, nomic-embed-text
- [ ] Chunk sizes: 800, 2500, 1500 (factual, technical, summary)
- [ ] Validation thresholds set appropriately

### Database Check
- [ ] `chroma_db/` directory created
- [ ] After indexing, `chroma_db/` contains files
- [ ] Can clear and reindex without errors

## 🚨 Troubleshooting Checklist

If errors occur, check:

### Ollama Issues
- [ ] `ollama serve` running in separate terminal
- [ ] Models downloaded: `ollama list`
- [ ] Can query model: `ollama run llama3 "test"`

### Python Issues
- [ ] Virtual environment activated (if using one)
- [ ] All dependencies installed: `pip list`
- [ ] Python version: `python3 --version`
- [ ] spaCy model: `python -c "import spacy; spacy.load('en_core_web_sm')"`

### Import Issues
- [ ] Run from correct directory: `cd geothermal-rag`
- [ ] Python path includes project: `echo $PYTHONPATH`
- [ ] Test imports: `python test_system.py`

### Performance Issues
- [ ] Sufficient RAM available: `free -h` (Linux) or Activity Monitor (Mac)
- [ ] No other heavy processes running
- [ ] Models cached after first query (subsequent queries faster)

## 📊 Success Criteria

System is ready for production use when:
- [x] All installation steps completed
- [x] Test suite passes (all ✓)
- [x] Application launches without errors
- [x] Can upload and index documents
- [x] Can query in all three modes
- [x] Extraction produces reasonable results
- [x] Validation catches errors appropriately
- [x] No critical errors in logs

## 📝 Deployment Notes

Record the following for reference:

**System Information:**
- Python version: _______________
- Ollama version: _______________
- Operating system: _______________
- RAM available: _______________

**Performance Baseline:**
- Time to index 27-page PDF: _______________
- Time for Q&A query: _______________
- Time for extraction: _______________

**Test Results:**
- Test PDF used: _______________
- Well name detected: _______________
- Extraction confidence: _______________
- Trajectory points found: _______________
- Validation passed: ☐ Yes ☐ No

**Known Issues:**
_________________________________________
_________________________________________
_________________________________________

**Date deployed:** _______________
**Deployed by:** _______________

---

## 🎉 Ready for Production!

Once all items are checked, the system is ready for:
- Real-world geothermal well report analysis
- Production capacity estimation
- Batch document processing
- Integration with existing workflows

## 📞 Support Resources

- **Documentation**: README.md
- **Quick start**: QUICKSTART.md  
- **Project details**: PROJECT_SUMMARY.md
- **Implementation notes**: context.txt (in parent directory)
- **Tests**: `python test_system.py`

---

**Last updated**: November 18, 2025
