#!/bin/bash

# Setup script for RAG Geothermal Wells system
# Run this after cloning the repository

set -e  # Exit on error

echo "=========================================="
echo "RAG for Geothermal Wells - Setup"
echo "=========================================="
echo ""

# Get the project root directory (parent of geothermal-rag)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '(?<=Python )\d+\.\d+')
REQUIRED_VERSION="3.9"

# Compare versions using bash (convert to integers: 3.9 -> 309, 3.12 -> 312)
PYTHON_INT=$(echo "$PYTHON_VERSION" | awk -F. '{printf "%d%02d", $1, $2}')
REQUIRED_INT=$(echo "$REQUIRED_VERSION" | awk -F. '{printf "%d%02d", $1, $2}')

if [ "$PYTHON_INT" -ge "$REQUIRED_INT" ]; then
    echo "✓ Python $PYTHON_VERSION detected (>= $REQUIRED_VERSION required)"
else
    echo "✗ Python $PYTHON_VERSION is too old. Please install Python >= $REQUIRED_VERSION"
    exit 1
fi

# Create or activate virtual environment
echo ""
echo "Setting up virtual environment..."
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"
echo "✓ Virtual environment activated"

# Upgrade pip in the virtual environment
echo "Upgrading pip..."
pip install --upgrade pip -q

# Check if Ollama is installed
echo ""
echo "Checking Ollama installation..."
OLLAMA_INSTALLED=false
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    OLLAMA_INSTALLED=true
else
    echo "⚠️  Ollama not found. Install from https://ollama.ai/ for full RAG features"
    echo "   (Setup will continue - core extraction/analysis will work without Ollama)"
fi

# Check if Ollama is running
OLLAMA_RUNNING=false
if [ "$OLLAMA_INSTALLED" = true ]; then
    echo ""
    echo "Checking if Ollama is running..."
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✓ Ollama is running"
        OLLAMA_RUNNING=true
    else
        echo "⚠️  Ollama is not running. Start it with: ollama serve"
        echo "   (Setup will continue, you can start it later)"
    fi
fi

# Install Python dependencies in the virtual environment
echo ""
echo "Installing Python dependencies in virtual environment..."
cd "$PROJECT_ROOT/geothermal-rag"
pip install -r requirements.txt

# Download spaCy model to the virtual environment
echo ""
echo "Downloading spaCy language model to virtual environment..."
python -m spacy download en_core_web_sm

# Pull Ollama models
if [ "$OLLAMA_INSTALLED" = true ]; then
    echo ""
    echo "Pulling Ollama models (Advanced RAG System, ~10 minutes)..."
    
    if [ "$OLLAMA_RUNNING" = true ]; then
        echo "Pulling llama3.1:8b (4.7GB - reasoning and QA)..."
        ollama pull llama3.1:8b
        
        echo "Pulling llava:7b (4.7GB - vision model for images)..."
        ollama pull llava:7b
        
        echo "Pulling nomic-embed-text (embeddings - optional fallback)..."
        ollama pull nomic-embed-text
        
        echo "✓ Ollama models downloaded"
        echo ""
        echo "NOTE: System uses sentence-transformers (all-MiniLM-L6-v2) for embeddings by default"
    else
        echo "⚠️  Ollama not running. Models not downloaded."
        echo "   Start Ollama later and run:"
        echo "   ollama pull llama3.1:8b"
        echo "   ollama pull llava:7b"
        echo "   ollama pull nomic-embed-text"
    fi
fi

# Create directories
echo ""
echo "Creating necessary directories..."
mkdir -p "$PROJECT_ROOT/geothermal-rag/chroma_db"
echo "✓ Created chroma_db directory"

# Run tests
echo ""
echo "Running system tests..."
cd "$PROJECT_ROOT/geothermal-rag"
python test_system.py

echo ""
echo "=========================================="
echo "Setup completed successfully! ✓"
echo "=========================================="
echo ""

if [ "$OLLAMA_INSTALLED" = false ]; then
    echo "⚠️  OLLAMA NOT INSTALLED - Limited functionality:"
    echo ""
    echo "   ✓ Core extraction/analysis works (run: python demo.py)"
    echo "   ✗ RAG features disabled (PDF indexing, semantic search, LLM Q&A)"
    echo ""
    echo "   To enable full functionality:"
    echo "   1. Install Ollama: https://ollama.ai/"
    echo "   2. Start Ollama: ollama serve"
    echo "   3. Pull models: ollama pull llama3.1:8b && ollama pull llava:7b"
    echo "   4. Run app: python app.py"
    echo ""
elif [ "$OLLAMA_RUNNING" = false ]; then
    echo "⚠️  OLLAMA NOT RUNNING - Start it before using the app:"
    echo "   ollama serve"
    echo ""
fi

echo "Virtual environment location: $VENV_DIR"
echo ""
echo "To use the application, activate the virtual environment first:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run:"
echo "  Core functionality test (no Ollama): python demo.py"
echo "  Full application (requires Ollama): python app.py"
echo ""
echo "The Gradio UI will be available at:"
echo "  http://localhost:7860"
echo ""
echo ""
echo "=========================================="
echo "Starting the application..."
echo "=========================================="
echo ""

# Start the main application
cd "$PROJECT_ROOT/geothermal-rag"
python app.py
