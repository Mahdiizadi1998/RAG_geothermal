#!/bin/bash
# Force clean and restart script for Linux/Mac

echo "=========================================="
echo "FORCE CLEAN AND RESTART"
echo "=========================================="
echo ""

echo "[1/5] Deleting old database..."
if [ -f well_data.db ]; then
    rm -f well_data.db
    echo "  ✓ Deleted well_data.db"
else
    echo "  - No database file found"
fi
echo ""

echo "[2/5] Deleting Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
echo "  ✓ Deleted __pycache__ directories"
echo ""

echo "[3/5] Deleting .pyc files..."
find . -type f -name "*.pyc" -delete 2>/dev/null
echo "  ✓ Deleted .pyc files"
echo ""

echo "[4/5] Verifying database_manager.py has correct schema..."
if grep -q "pipe_id_nominal" agents/database_manager.py; then
    echo "  ✓ pipe_id_nominal found in database_manager.py"
else
    echo "  ✗ ERROR: pipe_id_nominal NOT found in database_manager.py"
    echo "  Your file may be outdated! Pull latest changes."
    exit 1
fi
echo ""

echo "[5/5] All clean! Ready to start."
echo ""
echo "=========================================="
echo "Next steps:"
echo "  1. Run: python app.py"
echo "  2. Upload your PDF"
echo "  3. Database will be created with NEW schema"
echo "=========================================="
echo ""
