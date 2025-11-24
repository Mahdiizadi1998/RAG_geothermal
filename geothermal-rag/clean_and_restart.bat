@echo off
REM Force clean and restart script for Windows

echo ==========================================
echo FORCE CLEAN AND RESTART
echo ==========================================
echo.

echo [1/5] Deleting old database...
if exist well_data.db (
    del /F well_data.db
    echo   - Deleted well_data.db
) else (
    echo   - No database file found
)
echo.

echo [2/5] Deleting Python cache files...
if exist __pycache__ (
    rmdir /S /Q __pycache__
    echo   - Deleted __pycache__
)
if exist agents\__pycache__ (
    rmdir /S /Q agents\__pycache__
    echo   - Deleted agents\__pycache__
)
if exist models\__pycache__ (
    rmdir /S /Q models\__pycache__
    echo   - Deleted models\__pycache__
)
if exist utils\__pycache__ (
    rmdir /S /Q utils\__pycache__
    echo   - Deleted utils\__pycache__
)
echo.

echo [3/5] Deleting .pyc files...
del /S /Q *.pyc 2>nul
echo   - Deleted .pyc files
echo.

echo [4/5] Verifying database_manager.py has correct schema...
findstr /C:"pipe_id_nominal" agents\database_manager.py >nul
if %ERRORLEVEL% EQU 0 (
    echo   - [OK] pipe_id_nominal found in database_manager.py
) else (
    echo   - [ERROR] pipe_id_nominal NOT found in database_manager.py
    echo   - Your file may be outdated! Pull latest changes.
    pause
    exit /b 1
)
echo.

echo [5/5] All clean! Ready to start.
echo.
echo ==========================================
echo Next steps:
echo   1. Run: python app.py
echo   2. Upload your PDF
echo   3. Database will be created with NEW schema
echo ==========================================
echo.
pause
