@echo off
REM Simple diagnostic script

echo ==========================================
echo Environment Diagnostic
echo ==========================================
echo.

REM Check PowerShell version
echo PowerShell Version:
powershell -Command "$PSVersionTable.PSVersion"
echo.

REM Check Python
echo Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check 'Add Python to PATH' during installation
) else (
    for /f "tokens=*" %%i in ('python --version 2^>^&1') do echo [OK] %%i
    for /f "tokens=*" %%i in ('where python') do echo     Location: %%i
)
echo.

REM Check pip
echo Checking pip...
python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not available
) else (
    for /f "tokens=*" %%i in ('python -m pip --version 2^>^&1') do echo [OK] %%i
)
echo.

REM Check current directory
echo Current Directory:
echo %CD%
echo.

REM Check script location
echo Script Directory:
echo %~dp0
echo.

REM Check virtual environment
set "PROJECT_ROOT=%~dp0.."
set "VENV_DIR=%PROJECT_ROOT%\.venv"
echo Expected Virtual Environment Location:
echo %VENV_DIR%
if exist "%VENV_DIR%" (
    echo [OK] Directory exists
) else (
    echo [INFO] Directory does not exist ^(will be created during setup^)
)
echo.

REM Check Ollama
echo Checking Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Ollama not found ^(optional - needed for RAG features^)
    echo Install from: https://ollama.ai/
) else (
    for /f "tokens=*" %%i in ('where ollama') do echo [OK] Ollama found at: %%i
    
    REM Check if running
    curl -s http://localhost:11434/api/tags >nul 2>&1
    if errorlevel 1 (
        echo [WARNING] Ollama is installed but not running
        echo Start it with: ollama serve
    ) else (
        echo [OK] Ollama is running
    )
)
echo.

echo ==========================================
echo Diagnostic Complete
echo ==========================================
echo.
pause
