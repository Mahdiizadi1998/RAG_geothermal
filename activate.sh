#!/bin/bash

# Convenience script to activate the virtual environment
# Usage: source activate.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "Run setup first: cd geothermal-rag && bash setup.sh"
    return 1
fi

source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"
echo ""
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "To run the application:"
echo "  cd geothermal-rag"
echo "  python demo.py  (core features, no Ollama needed)"
echo "  python app.py   (full RAG features, requires Ollama)"
