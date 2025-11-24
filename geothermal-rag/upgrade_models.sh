#!/bin/bash
# Setup script for improved summarization system

echo "========================================"
echo "Summarization System Upgrade"
echo "========================================"
echo

echo "Installing upgraded models for better accuracy..."
echo

# Check if ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama is not running!"
    echo "Please start Ollama first: ollama serve"
    exit 1
fi

echo "✓ Ollama is running"
echo

# Install llama3.1:8b for QA, Summary, and Verification
echo "📦 Installing llama3.1:8b (4.7GB) for QA, Summary, and Verification..."
ollama pull llama3.1:8b
if [ $? -eq 0 ]; then
    echo "✓ llama3.1:8b installed successfully"
else
    echo "❌ Failed to install llama3.1:8b"
    exit 1
fi
echo

# Install qwen2.5:14b for Extraction
echo "📦 Installing qwen2.5:14b (8.7GB) for high-accuracy extraction..."
echo "   (This may take a while...)"
ollama pull qwen2.5:14b
if [ $? -eq 0 ]; then
    echo "✓ qwen2.5:14b installed successfully"
else
    echo "❌ Failed to install qwen2.5:14b"
    echo "   If this is too large, you can use qwen2.5:7b instead"
    echo "   Edit config.yaml: model_extraction: qwen2.5:7b"
    exit 1
fi
echo

# Verify installations
echo "========================================"
echo "Verifying installations..."
echo "========================================"
ollama list
echo

echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo
echo "✓ Models installed:"
echo "  - llama3.1:8b (QA, Summary, Verification)"
echo "  - qwen2.5:14b (Extraction)"
echo
echo "📝 Next steps:"
echo "  1. Clear old index in Gradio UI (different chunk sizes)"
echo "  2. Re-upload your PDFs"
echo "  3. Generate a summary and check for:"
echo "     - Citations: [Source: filename, Page X]"
echo "     - Fact verification: >90%"
echo "     - Correct data (depths, casing specs)"
echo "     - High confidence: >85%"
echo
echo "🚀 Ready to generate accurate summaries with citations!"
