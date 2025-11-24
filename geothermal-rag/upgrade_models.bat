@echo off
REM Setup script for improved summarization system (Windows)

echo ========================================
echo Summarization System Upgrade
echo ========================================
echo.

echo Installing upgraded models for better accuracy...
echo.

REM Check if ollama is running
curl -s http://localhost:11434/api/tags >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo X Ollama is not running!
    echo Please start Ollama first: ollama serve
    pause
    exit /b 1
)

echo [OK] Ollama is running
echo.

REM Install llama3.1:8b for QA, Summary, and Verification
echo Installing llama3.1:8b (4.7GB) for QA, Summary, and Verification...
ollama pull llama3.1:8b
if %ERRORLEVEL% EQU 0 (
    echo [OK] llama3.1:8b installed successfully
) else (
    echo [ERROR] Failed to install llama3.1:8b
    pause
    exit /b 1
)
echo.

REM Install qwen2.5:14b for Extraction
echo Installing qwen2.5:14b (8.7GB) for high-accuracy extraction...
echo (This may take a while...)
ollama pull qwen2.5:14b
if %ERRORLEVEL% EQU 0 (
    echo [OK] qwen2.5:14b installed successfully
) else (
    echo [ERROR] Failed to install qwen2.5:14b
    echo If this is too large, you can use qwen2.5:7b instead
    echo Edit config.yaml: model_extraction: qwen2.5:7b
    pause
    exit /b 1
)
echo.

REM Verify installations
echo ========================================
echo Verifying installations...
echo ========================================
ollama list
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo [OK] Models installed:
echo   - llama3.1:8b (QA, Summary, Verification)
echo   - qwen2.5:14b (Extraction)
echo.
echo Next steps:
echo   1. Clear old index in Gradio UI (different chunk sizes)
echo   2. Re-upload your PDFs
echo   3. Generate a summary and check for:
echo      - Citations: [Source: filename, Page X]
echo      - Fact verification: ^>90%%
echo      - Correct data (depths, casing specs)
echo      - High confidence: ^>85%%
echo.
echo Ready to generate accurate summaries with citations!
echo.
pause
