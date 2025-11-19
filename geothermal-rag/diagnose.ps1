# Diagnostic script to check environment
# Run this first to see what's working

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Environment Diagnostic" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check PowerShell version
Write-Host "PowerShell Version:"
$PSVersionTable.PSVersion
Write-Host ""

# Check Python
Write-Host "Checking Python..."
try {
    $pythonPath = (Get-Command python -ErrorAction Stop).Source
    Write-Host "✓ Python found at: $pythonPath" -ForegroundColor Green
    $pythonVersion = & python --version 2>&1
    Write-Host "  Version: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found in PATH" -ForegroundColor Red
    Write-Host "  Please install Python from https://www.python.org/downloads/" -ForegroundColor Yellow
    Write-Host "  Make sure to check 'Add Python to PATH' during installation" -ForegroundColor Yellow
}
Write-Host ""

# Check pip
Write-Host "Checking pip..."
try {
    $pipVersion = & python -m pip --version 2>&1
    Write-Host "✓ pip: $pipVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ pip not available" -ForegroundColor Red
}
Write-Host ""

# Check current directory
Write-Host "Current Directory:"
Write-Host "  $PWD"
Write-Host ""

# Check script location
Write-Host "Script Directory:"
Write-Host "  $PSScriptRoot"
Write-Host ""

# Check if virtual environment exists
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
$VENV_DIR = Join-Path $PROJECT_ROOT ".venv"
Write-Host "Expected Virtual Environment Location:"
Write-Host "  $VENV_DIR"
if (Test-Path $VENV_DIR) {
    Write-Host "  ✓ Directory exists" -ForegroundColor Green
} else {
    Write-Host "  ✗ Directory does not exist (will be created during setup)" -ForegroundColor Yellow
}
Write-Host ""

# Check Ollama
Write-Host "Checking Ollama..."
try {
    $ollamaPath = (Get-Command ollama -ErrorAction Stop).Source
    Write-Host "✓ Ollama found at: $ollamaPath" -ForegroundColor Green
    
    # Check if running
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -TimeoutSec 2 -ErrorAction Stop
        Write-Host "✓ Ollama is running" -ForegroundColor Green
    } catch {
        Write-Host "⚠️  Ollama is installed but not running" -ForegroundColor Yellow
        Write-Host "  Start it with: ollama serve" -ForegroundColor Yellow
    }
} catch {
    Write-Host "⚠️  Ollama not found (optional - needed for RAG features)" -ForegroundColor Yellow
    Write-Host "  Install from: https://ollama.ai/" -ForegroundColor Yellow
}
Write-Host ""

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Diagnostic Complete" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press any key to exit..." -ForegroundColor Yellow
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
