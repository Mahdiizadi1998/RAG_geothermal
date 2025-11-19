# Project Structure

## Directory Layout

```
/workspaces/RAG_geothermal/           # PROJECT ROOT
├── .venv/                            # Virtual environment (all dependencies here)
│   ├── bin/                          # Python executables
│   ├── lib/                          # Installed packages
│   └── pyvenv.cfg                    # venv configuration
│
├── geothermal-rag/                   # Main application directory
│   ├── agents/                       # AI agent modules
│   │   ├── __init__.py
│   │   ├── ingestion_agent.py
│   │   ├── preprocessing_agent.py
│   │   ├── rag_retrieval_agent.py
│   │   ├── parameter_extraction_agent.py
│   │   ├── validation_agent.py
│   │   ├── ensemble_judge_agent.py
│   │   └── chat_memory.py
│   │
│   ├── models/                       # Analysis models
│   │   ├── __init__.py
│   │   └── nodal_analysis.py
│   │
│   ├── utils/                        # Utility modules
│   │   ├── __init__.py
│   │   ├── pattern_library.py
│   │   └── unit_conversion.py
│   │
│   ├── config/                       # Configuration files
│   │   └── config.yaml
│   │
│   ├── chroma_db/                    # Vector database storage
│   │
│   ├── app.py                        # Main Gradio application
│   ├── demo.py                       # Demo script (no Ollama needed)
│   ├── test_system.py                # Test suite
│   ├── setup.sh                      # Automated setup script
│   ├── requirements.txt              # Python dependencies
│   │
│   └── *.md                          # Documentation files
│
├── activate.sh                       # Convenience script to activate venv
└── README.md                         # Project documentation

```

## Key Points

### ✅ Everything Self-Contained
- **Virtual environment**: `.venv/` in project root
- **Dependencies**: Installed in `.venv/lib/python3.12/site-packages/`
- **Vector DB**: `chroma_db/` in application directory
- **Models**: spaCy model in `.venv/`
- **No global installations**: Everything isolated to this project

### 🚀 Quick Commands

**Initial Setup:**
```bash
cd /workspaces/RAG_geothermal/geothermal-rag
bash setup.sh
```

**Activate Environment:**
```bash
source /workspaces/RAG_geothermal/.venv/bin/activate
# OR
source /workspaces/RAG_geothermal/activate.sh
```

**Run Applications:**
```bash
cd /workspaces/RAG_geothermal/geothermal-rag
python demo.py   # Core features (no Ollama)
python app.py    # Full RAG (requires Ollama)
```

**Deactivate Environment:**
```bash
deactivate
```

### 📦 What's Installed Where

| Component | Location |
|-----------|----------|
| Python packages | `.venv/lib/python3.12/site-packages/` |
| Python executable | `.venv/bin/python` |
| pip | `.venv/bin/pip` |
| spaCy model | `.venv/lib/python3.12/site-packages/en_core_web_sm/` |
| Application code | `geothermal-rag/` |
| Vector database | `geothermal-rag/chroma_db/` |
| Configuration | `geothermal-rag/config/config.yaml` |

### 🔍 Verification

**Check virtual environment:**
```bash
source /workspaces/RAG_geothermal/.venv/bin/activate
which python
# Should output: /workspaces/RAG_geothermal/.venv/bin/python
```

**Check installed packages:**
```bash
pip list
```

**Check spaCy model:**
```bash
python -c "import spacy; print(spacy.load('en_core_web_sm'))"
```

### 🎯 Benefits

1. **Isolated**: No system-wide package pollution
2. **Portable**: Entire project in one directory
3. **Reproducible**: Same environment on any machine
4. **Clean**: Easy to delete (just remove `.venv/`)
5. **Safe**: No conflicts with other Python projects

### 🗑️ Clean Uninstall

To completely remove all installed dependencies:
```bash
cd /workspaces/RAG_geothermal
rm -rf .venv
rm -rf geothermal-rag/chroma_db
```

Then re-run `setup.sh` to reinstall if needed.
