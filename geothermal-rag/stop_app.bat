@echo off
REM Script to stop existing Gradio instance on port 7860

echo ========================================
echo Stopping Gradio on Port 7860
echo ========================================
echo.

echo Checking for processes using port 7860...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :7860') do (
    set PID=%%a
)

if defined PID (
    echo Found process with PID: %PID%
    echo Attempting to terminate...
    taskkill /PID %PID% /F
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ========================================
        echo Successfully stopped process
        echo ========================================
        echo.
        echo You can now run: python app.py
    ) else (
        echo.
        echo ========================================
        echo Failed to stop process
        echo ========================================
        echo Try running this script as Administrator
    )
) else (
    echo.
    echo ========================================
    echo No process found using port 7860
    echo ========================================
    echo Port is already available
)

echo.
pause
