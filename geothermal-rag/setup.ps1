# Setup script for RAG Geothermal Wells system - PowerShell
# Run this after cloning the repository

# Error handling
$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "RAG for Geothermal Wells - Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

try {
    # Get the project root directory (parent of geothermal-rag)
    $PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
    $VENV_DIR = Join-Path $PROJECT_ROOT ".venv"

    Write-Host "Project root: $PROJECT_ROOT"
    Write-Host ""

    # Check Python version
    Write-Host "Checking Python version..."
    try {
        $pythonVersion = & python --version 2>&1 | Out-String
        Write-Host "Python found: $pythonVersion"
        
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            $versionNum = $major * 100 + $minor
            
            if ($versionNum -ge 309) {
                Write-Host "✓ Python $major.$minor detected (>= 3.9 required)" -ForegroundColor Green
            } else {
                throw "Python $major.$minor is too old. Please install Python >= 3.9"
            }
        } else {
            throw "Could not parse Python version"
        }
    } catch {
        Write-Host "✗ Python not found or incompatible" -ForegroundColor Red
        Write-Host "Please install Python >= 3.9 from https://www.python.org/downloads/" -ForegroundColor Yellow
        Write-Host "Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
        throw
    }

    # Create or check virtual environment
    Write-Host ""
    Write-Host "Setting up virtual environment..."
    if (-Not (Test-Path $VENV_DIR)) {
        Write-Host "Creating virtual environment at $VENV_DIR..."
        & python -m venv $VENV_DIR
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create virtual environment"
        }
        Write-Host "✓ Virtual environment created" -ForegroundColor Green
    } else {
        Write-Host "✓ Virtual environment already exists" -ForegroundColor Green
    }

    # Activate virtual environment
    $activateScript = Join-Path $VENV_DIR "Scripts\Activate.ps1"
    if (Test-Path $activateScript) {
        Write-Host "Activating virtual environment..."
        & $activateScript
        Write-Host "✓ Virtual environment activated" -ForegroundColor Green
    } else {
        throw "Cannot find activation script at: $activateScript"
    }

    # Upgrade pip
    Write-Host ""
    Write-Host "Upgrading pip..."
    & python -m pip install --upgrade pip --quiet
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ pip upgraded" -ForegroundColor Green
    }

    # Check if Ollama is installed
    Write-Host ""
    Write-Host "Checking Ollama installation..."
    $OLLAMA_INSTALLED = $false
    $OLLAMA_RUNNING = $false
    
    try {
        $null = Get-Command ollama -ErrorAction Stop
        Write-Host "✓ Ollama is installed" -ForegroundColor Green
        $OLLAMA_INSTALLED = $true
        
        # Check if running
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
            Write-Host "✓ Ollama is running" -ForegroundColor Green
            $OLLAMA_RUNNING = $true
        } catch {
            Write-Host "⚠️  Ollama is not running" -ForegroundColor Yellow
        }
    } catch {
        Write-Host "⚠️  Ollama not found (optional)" -ForegroundColor Yellow
    }

    # Install Python dependencies
    Write-Host ""
    Write-Host "Installing Python dependencies..."
    Set-Location (Join-Path $PROJECT_ROOT "geothermal-rag")
    & python -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python dependencies"
    }
    Write-Host "✓ Dependencies installed" -ForegroundColor Green

    # Download spaCy model
    Write-Host ""
    Write-Host "Downloading spaCy language model..."
    & python -m spacy download en_core_web_sm
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ spaCy model downloaded" -ForegroundColor Green
    }

    # Pull Ollama models
    if ($OLLAMA_INSTALLED -and $OLLAMA_RUNNING) {
        Write-Host ""
        Write-Host "Pulling Ollama models (this may take a few minutes)..."
        
        Write-Host "Pulling llama3..."
        & ollama pull llama3
        
        Write-Host "Pulling nomic-embed-text..."
        & ollama pull nomic-embed-text
        
        Write-Host "✓ Ollama models downloaded" -ForegroundColor Green
    }

    # Create directories
    Write-Host ""
    Write-Host "Creating necessary directories..."
    $chromaDbPath = Join-Path $PROJECT_ROOT "geothermal-rag\chroma_db"
    New-Item -ItemType Directory -Force -Path $chromaDbPath | Out-Null
    Write-Host "✓ Created chroma_db directory" -ForegroundColor Green

    # Run tests
    Write-Host ""
    Write-Host "Running system tests..."
    Set-Location (Join-Path $PROJECT_ROOT "geothermal-rag")
    & python test_system.py
    if ($LASTEXITCODE -ne 0) {
        Write-Host "⚠️  Tests failed but continuing..." -ForegroundColor Yellow
    } else {
        Write-Host "✓ All tests passed" -ForegroundColor Green
    }

    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Setup completed successfully! ✓" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""

    if (-Not $OLLAMA_INSTALLED) {
        Write-Host "⚠️  OLLAMA NOT INSTALLED - Limited functionality" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "   ✓ Core extraction/analysis works"
        Write-Host "   ✗ RAG features disabled"
        Write-Host ""
        Write-Host "   To enable full functionality:"
        Write-Host "   1. Install Ollama: https://ollama.ai/"
        Write-Host "   2. Start Ollama: ollama serve"
        Write-Host "   3. Pull models: ollama pull llama3 && ollama pull nomic-embed-text"
        Write-Host ""
    } elseif (-Not $OLLAMA_RUNNING) {
        Write-Host "⚠️  Start Ollama before using RAG features: ollama serve" -ForegroundColor Yellow
        Write-Host ""
    }

    Write-Host "Virtual environment: $VENV_DIR"
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host "Starting the application..." -ForegroundColor Cyan
    Write-Host "==========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "The Gradio UI will be available at: http://localhost:7860" -ForegroundColor Green
    Write-Host ""

    # Start the main application
    Set-Location (Join-Path $PROJECT_ROOT "geothermal-rag")
    & python app.py

} catch {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "Setup Failed!" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Error: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting steps:" -ForegroundColor Yellow
    Write-Host "1. Make sure Python >= 3.9 is installed and in PATH"
    Write-Host "2. Run 'diagnose.bat' to check your environment"
    Write-Host "3. Check if antivirus is blocking Python/pip"
    Write-Host "4. Try running as Administrator"
    Write-Host ""
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit 1
}

# Pause if there was an error with the app
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Application exited with errors" -ForegroundColor Yellow
    Write-Host "Press any key to exit..." -ForegroundColor Yellow
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
