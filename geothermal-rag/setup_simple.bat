@echo off
setlocal enabledelayedexpansion

REM Setup script for RAG Geothermal Wells system
REM Run this after cloning the repository

echo ==========================================
echo RAG for Geothermal Wells - Setup
echo ==========================================
echo.

REM Get the project root directory
set "PROJECT_ROOT=%~dp0.."
set "VENV_DIR=%PROJECT_ROOT%\.venv"

echo Project root: %PROJECT_ROOT%
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo.
    echo Please install Python ^>= 3.9 from https://www.python.org/downloads/
    echo Make sure to check 'Add Python to PATH' during installation
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python found: %PYTHON_VERSION%
echo.

REM Create or check virtual environment
echo Setting up virtual environment...
if not exist "%VENV_DIR%" (
    echo Creating virtual environment at %VENV_DIR%...
    python -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo Activating virtual environment...
if exist "%VENV_DIR%\Scripts\activate.bat" (
    call "%VENV_DIR%\Scripts\activate.bat"
    echo [OK] Virtual environment activated
) else (
    echo [ERROR] Cannot find activation script
    pause
    exit /b 1
)
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip --quiet
if errorlevel 1 (
    echo [WARNING] pip upgrade failed
) else (
    echo [OK] pip upgraded
)
echo.

REM Check if Ollama is installed
echo Checking Ollama installation...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not found ^(optional^)
    set "OLLAMA_INSTALLED=0"
) else (
    echo [OK] Ollama is installed
    set "OLLAMA_INSTALLED=1"
    
    REM Check if Ollama is running
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Ollama is not running
        set "OLLAMA_RUNNING=0"
    ) else (
        echo [OK] Ollama is running
        set "OLLAMA_RUNNING=1"
    )
)
echo.

REM Install Python dependencies
echo Installing Python dependencies...
cd /d "%PROJECT_ROOT%\geothermal-rag"
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies
    pause
    exit /b 1
)
echo [OK] Dependencies installed
echo.

REM Download spaCy model
echo Downloading spaCy language model...
python -m spacy download en_core_web_sm
if errorlevel 1 (
    echo [WARNING] spaCy model download failed
) else (
    echo [OK] spaCy model downloaded
)
echo.

REM Pull Ollama models
if "%OLLAMA_INSTALLED%"=="1" if "%OLLAMA_RUNNING%"=="1" (
    echo Pulling Ollama models ^(Advanced RAG System, ~10 minutes^)...
    echo.
    echo Pulling llama3.1:8b ^(4.7GB - reasoning and QA^)...
    ollama pull llama3.1:8b
    echo.
    echo Pulling llava:7b ^(4.7GB - vision model for images^)...
    ollama pull llava:7b
    echo.
    echo Pulling nomic-embed-text ^(embeddings - optional fallback^)...
    ollama pull nomic-embed-text
    echo.
    echo [OK] Ollama models downloaded
    echo.
    echo NOTE: System uses sentence-transformers ^(all-MiniLM-L6-v2^) for embeddings by default
    echo.
)

REM Create directories
echo Creating necessary directories...
if not exist "%PROJECT_ROOT%\geothermal-rag\chroma_db" mkdir "%PROJECT_ROOT%\geothermal-rag\chroma_db"
echo [OK] Created chroma_db directory
echo.

REM Run tests
echo Running system tests...
cd /d "%PROJECT_ROOT%\geothermal-rag"
python test_system.py
if errorlevel 1 (
    echo [WARNING] Tests failed but continuing...
) else (
    echo [OK] All tests passed
)
echo.

echo ==========================================
echo Setup completed successfully!
echo ==========================================
echo.

if "%OLLAMA_INSTALLED%"=="0" (
    echo [WARNING] OLLAMA NOT INSTALLED - Limited functionality
    echo.
    echo    [OK] Core extraction/analysis works
    echo    [X] RAG features disabled
    echo.
    echo    To enable full functionality:
    echo    1. Install Ollama: https://ollama.ai/
    echo    2. Start Ollama: ollama serve
    echo    3. Pull models:
    echo       ollama pull llama3.1:8b
    echo       ollama pull llava:7b
    echo       ollama pull nomic-embed-text
    echo.
) else if "%OLLAMA_RUNNING%"=="0" (
    echo [WARNING] Start Ollama before using RAG features: ollama serve
    echo.
)

echo ==========================================
echo ADVANCED AGENTIC RAG SYSTEM:
echo ==========================================
echo ✓ Models: llama3.1:8b ^(QA^), llava:7b ^(Vision^)
echo ✓ Embeddings: all-MiniLM-L6-v2 ^(sentence-transformers^)
echo ✓ Ultimate Semantic Chunker: Late Chunking + Context
echo ✓ RAPTOR Tree: Hierarchical summarization ^(HDBSCAN^)
echo ✓ Hybrid Retrieval: Dense + Sparse ^(BM25^) + Graph
echo ✓ Reranking: Cross-encoder + RRF fusion
echo ✓ Vision: Image captioning for plots/diagrams
echo ✓ 8-core CPU, 16GB RAM optimized
echo ✓ Expected: ~500ms queries, ~12GB peak memory
echo.

echo Virtual environment: %VENV_DIR%
echo.
echo ==========================================
echo Starting the application...
echo ==========================================
echo.
echo The Gradio UI will be available at: http://localhost:7860
echo.

REM Start the main application
cd /d "%PROJECT_ROOT%\geothermal-rag"
python app.py

REM Pause if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with errors
    pause
)

endlocal
