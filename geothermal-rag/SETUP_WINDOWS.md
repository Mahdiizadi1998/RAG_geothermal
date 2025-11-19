# RAG for Geothermal Wells - Windows Setup Guide

## Quick Start

You have two options to run the setup:

### Option 1: Double-click `setup.bat` (Recommended)
This bypasses execution policy restrictions automatically.

### Option 2: Allow PowerShell scripts permanently
Run this command in PowerShell as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
Then run:
```powershell
.\setup.ps1
```

## Troubleshooting

### Run Diagnostic First
Double-click `diagnose.bat` to check your environment before setup.

### Common Issues

**"Python not found"**
1. Install Python from https://www.python.org/downloads/
2. During installation, CHECK the box: "Add Python to PATH"
3. Restart PowerShell/Command Prompt
4. Run `diagnose.bat` again to verify

**"Execution Policy Error"**
- Use `setup.bat` instead of `setup.ps1` directly

**Window closes too fast**
- Check `setup_log.txt` in the geothermal-rag folder for error details

**Setup completes but app doesn't start**
- Press Ctrl+C to stop if stuck
- Check if Python dependencies installed correctly
- Try running manually:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  cd geothermal-rag
  python app.py
  ```

## What the Setup Does

1. ✅ Checks Python version (requires >= 3.9)
2. ✅ Creates virtual environment
3. ✅ Installs all dependencies
4. ✅ Downloads spaCy language model
5. ✅ Checks for Ollama (optional)
6. ✅ Runs system tests
7. ✅ Starts the Gradio web interface

## After Setup

The Gradio UI will be available at: **http://localhost:7860**

## Manual Setup (if automated setup fails)

```powershell
# Create virtual environment
python -m venv ..\.venv

# Activate it
..\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run the app
python app.py
```

## Need Help?

Check the log file: `setup_log.txt`
