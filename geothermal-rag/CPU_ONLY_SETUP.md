# CPU-Only Setup for Geothermal RAG System

## Problem: CUDA Out of Memory Errors

If you see errors like:
```
CUDA error: out of memory
cudaMalloc failed: out of memory
ONNXRuntimeError: bad allocation
CUDA error: the requested functionality is not supported
```

Your GPU doesn't have enough VRAM (need 4GB+). **Solution: Run entirely on CPU.**

---

## Complete CPU-Only Setup (Windows)

### Step 1: Stop Ollama

Close Ollama completely:
1. Check system tray → Right-click Ollama icon → Exit
2. Or in Task Manager → End "Ollama" process

### Step 2: Configure CPU-Only Mode

**Option A: Permanent (Recommended)**
```cmd
# Windows Command Prompt (Run as Administrator)
setx OLLAMA_NUM_GPU 0 /M

# Restart your computer for system-wide effect
```

**Option B: Per-Session**
```cmd
# Windows Command Prompt
set OLLAMA_NUM_GPU=0
ollama serve
```

**Option C: Per-Session (PowerShell)**
```powershell
$env:OLLAMA_NUM_GPU=0
ollama serve
```

### Step 3: Verify CPU Mode

Open new terminal and check:
```cmd
# Should show CPU processing
ollama run mistral:7b "test"
```

If working correctly, you'll see slower but **stable** processing with no CUDA errors.

### Step 4: Restart Geothermal RAG App

```cmd
cd geothermal-rag
python app.py
```

---

## Complete CPU-Only Setup (Linux/macOS)

### Step 1: Stop Ollama
```bash
# Find and kill ollama process
pkill ollama
```

### Step 2: Configure CPU-Only Mode

**Option A: Permanent (Recommended)**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export OLLAMA_NUM_GPU=0' >> ~/.bashrc
source ~/.bashrc
```

**Option B: Per-Session**
```bash
export OLLAMA_NUM_GPU=0
ollama serve &
```

### Step 3: Verify
```bash
echo $OLLAMA_NUM_GPU  # Should print: 0
ollama run mistral:7b "test"
```

### Step 4: Restart App
```bash
cd geothermal-rag
python app.py
```

---

## Expected Performance (CPU Mode)

| Task | GPU Time | CPU Time | Notes |
|------|----------|----------|-------|
| Document Upload (3 PDFs) | 20-30 sec | 60-90 sec | 3x slower |
| Embeddings (100 chunks) | 5 sec | 15-20 sec | 3-4x slower |
| Q&A Query | 5-10 sec | 30-60 sec | 5-6x slower |
| Summary (200 words) | 3-5 sec | 20-40 sec | 6-8x slower |
| Extraction + Analysis | 30-60 sec | 3-5 min | 5-6x slower |

**Total workflow**: GPU ~2 min → CPU ~8-12 min

---

## Optimized Model Configuration (CPU)

Current config uses CPU-optimized models:

```yaml
# config/config.yaml
ollama:
  model_qa: mistral:7b          # Fast on CPU
  model_summary: gemma2:2b      # Very fast (2B params)
  model_verification: mistral:7b
  model_extraction: qwen2.5:7b  # Best accuracy
```

**Trade-offs**:
- ✅ No memory errors, stable
- ✅ Works on any hardware
- ❌ 5-8x slower than GPU
- ✅ Same quality results

---

## Alternative: Smaller Models (Even Faster CPU)

If still too slow, try:

```yaml
ollama:
  model_qa: phi3:mini           # 2.2GB, faster
  model_summary: tinyllama      # 637MB, very fast
  model_verification: phi3:mini
  model_extraction: qwen2.5:3b  # 1.9GB, smaller Qwen
```

Pull these models:
```bash
ollama pull phi3:mini
ollama pull tinyllama
ollama pull qwen2.5:3b
```

**Trade-offs**:
- ✅ 2-3x faster than 7B models on CPU
- ❌ Slightly lower quality
- ✅ Good for development/testing

---

## Troubleshooting

### "Still getting CUDA errors"
**Cause**: Ollama server wasn't restarted with CPU flag

**Fix**:
1. Completely close Ollama (system tray + Task Manager)
2. Set `OLLAMA_NUM_GPU=0`
3. Start Ollama fresh: `ollama serve`
4. Verify: Check Ollama logs for "CUDA" mentions

### "Models are extremely slow"
**Cause**: Large models (llama3.1: 4.9GB) on CPU

**Fix**: Use smaller models (see Alternative section above)

### "Embeddings failing"
**Cause**: ChromaDB trying to use ONNX (GPU-based)

**Fix**: Already configured! System uses Ollama embeddings (CPU-based)

### "App crashes during indexing"
**Cause**: PDF text extraction or chunking issues

**Fix**:
1. Check PDF quality (not scanned images)
2. Verify `nomic-embed-text` is running: `ollama pull nomic-embed-text`
3. Check logs for specific errors

---

## Verifying CPU-Only Operation

### Check 1: Ollama Logs
Look for:
```
✅ GOOD (CPU):
load_tensors: offloaded 0/33 layers to GPU
load_tensors: CPU model buffer size = 4685.30 MiB

❌ BAD (Still trying GPU):
load_tensors: offloaded 14/33 layers to GPU
cudaMalloc failed: out of memory
```

### Check 2: System Resource Monitor

**Windows**: Task Manager → Performance
- GPU usage should be **0-5%**
- CPU usage should be **60-100%** during processing

**Linux**: `htop` + `nvidia-smi`
- `nvidia-smi` should show **no ollama process**
- `htop` should show high CPU usage

### Check 3: App Logs
```
INFO:agents.rag_retrieval_agent:🖥️  Configured for CPU-only mode (OLLAMA_NUM_GPU=0)
INFO:agents.rag_retrieval_agent:📊 Using Ollama embeddings: nomic-embed-text
```

---

## FAQ

**Q: Will CPU-only work for production?**
A: Yes, but slower. Consider:
- Development: CPU is fine
- Demo/Testing: CPU is fine  
- Production (high volume): Get GPU with 6GB+ VRAM

**Q: Can I use both CPU and GPU?**
A: Yes! Set `OLLAMA_NUM_GPU=1` (uses 1 GPU layer). But with 3.3GB VRAM you'll still hit limits.

**Q: Which model size for CPU?**
A: Recommended hierarchy:
1. **3B models** (qwen2.5:3b, phi3:mini) - Best CPU performance
2. **7B models** (mistral, qwen2.5:7b) - Good balance
3. **8B+ models** (llama3, llama3.1) - Slow on CPU

**Q: How to speed up CPU processing?**
A: 
1. Use smaller models (3B instead of 7B)
2. Reduce `top_k` in config.yaml (fewer chunks to process)
3. Enable parallel processing (more CPU cores)
4. Close other applications

---

## Summary

✅ **CPU-only mode works perfectly** - just slower
✅ **No code changes needed** - just Ollama configuration
✅ **Same quality results** - models work identically
❌ **5-8x slower** than GPU mode
✅ **Stable** - no memory crashes

**Bottom line**: For your RTX 3050 (3.3GB VRAM), CPU-only is the reliable choice.
