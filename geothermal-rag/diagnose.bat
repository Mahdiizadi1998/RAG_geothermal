@echo off
REM Diagnostic launcher for RAG Geothermal Wells system

echo ==========================================
echo Environment Diagnostic Launcher
echo ==========================================
echo.

REM Run PowerShell diagnostic script with bypass execution policy
powershell.exe -ExecutionPolicy Bypass -File "%~dp0diagnose.ps1"
