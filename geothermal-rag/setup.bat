@echo off
REM Setup launcher for RAG Geothermal Wells system
REM This batch file bypasses execution policy restrictions

echo ==========================================
echo RAG for Geothermal Wells - Setup Launcher
echo ==========================================
echo.

REM Run PowerShell script with bypass execution policy
powershell.exe -ExecutionPolicy Bypass -File "%~dp0setup.ps1"

REM Check if PowerShell script failed
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Setup encountered an error. Check setup_log.txt for details.
    echo.
    pause
)
