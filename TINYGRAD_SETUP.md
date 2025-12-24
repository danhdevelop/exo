# Running EXO with Tinygrad on Linux ARM64

This guide explains how to run exo with the tinygrad engine on Linux ARM64 (or any platform where MLX is not available).

## Problem

MLX (the default inference engine) is not available on Linux ARM64. It only supports:
- macOS (Apple Silicon)
- Linux x86_64

## Solution

We've made the following changes to support tinygrad as an alternative inference engine:

### 1. Made MLX Optional

MLX is now an optional dependency. The base installation includes tinygrad instead.

### 2. Automatic Fallback

If MLX is not available, the runner will automatically fall back to tinygrad.

### 3. Manual Engine Selection

You can explicitly choose which engine to use via the `INFERENCE_ENGINE` environment variable.

## Installation & Usage

### Step 1: Sync the updated dependencies

```bash
cd /home/dn-nguyen/Workspace/exo_labs/exo/exo
uv sync
```

This will install tinygrad and skip MLX (since it's not available on your platform).

### Step 2: Run with tinygrad

```bash
# Option 1: Let it auto-detect (will use tinygrad since MLX isn't available)
uv run exo

# Option 2: Explicitly set tinygrad (recommended for clarity)
INFERENCE_ENGINE=tinygrad uv run exo
```

### Step 3: Verify it's using tinygrad

You should see in the logs:
```
Using tinygrad inference engine
```

## Configuration

The tinygrad engine supports these environment variables:

```bash
export INFERENCE_ENGINE=tinygrad  # Use tinygrad engine
export TEMPERATURE=0.85           # Sampling temperature (default: 0.85)
export TOP_K=25                   # Top-k sampling (default: 25)
export TOP_P=0.9                  # Nucleus sampling (default: 0.9)
export MAX_TOKENS=8192           # Max tokens to generate (default: 8192)
```

## Supported Models

Currently supports Llama-based models:
- Llama 1B, 3B, 8B, 70B variants
- Models from HuggingFace in safetensors or PyTorch format

## Performance Considerations

### CPU vs GPU
- **Tinygrad**: Runs on CPU by default on Linux ARM64
- **MLX**: Would use GPU on Apple Silicon (if available)

### Expected Performance
- Tinygrad on CPU will be slower than MLX on GPU
- Good for development and testing
- Consider using smaller models (1B, 3B) for better performance

### Optimization Tips
1. Use smaller models for faster inference
2. Reduce MAX_TOKENS if you don't need long outputs
3. Adjust temperature and sampling parameters for quality vs speed

## Troubleshooting

### Issue: "MLX is not available on this platform"

**This is expected and normal!** The runner will automatically fall back to tinygrad.

To suppress this message, explicitly set the engine:
```bash
INFERENCE_ENGINE=tinygrad uv run exo
```

### Issue: Import errors for tinygrad

Make sure dependencies are synced:
```bash
uv sync
```

### Issue: Model loading fails

Check that:
1. You have enough disk space for model downloads
2. You have internet connectivity for HuggingFace downloads
3. The model you're trying to use is Llama-based

### Issue: Out of memory

Try:
1. Using a smaller model (1B instead of 8B)
2. Reducing MAX_TOKENS
3. Closing other applications

## Files Changed

1. **pyproject.toml**:
   - Moved MLX to optional dependencies
   - Added tinygrad as required dependency
   - Added transformers for tokenization

2. **src/exo/worker/runner/runner.py**:
   - Added engine selection logic
   - Added graceful fallback from MLX to tinygrad
   - Made imports conditional

3. **src/exo/worker/engines/tinygrad/** (new):
   - Complete tinygrad engine implementation
   - Llama model support
   - Generation and warmup functions

## Next Steps

1. Try running exo with a small model:
   ```bash
   INFERENCE_ENGINE=tinygrad uv run exo
   ```

2. Test with a Llama-based model from HuggingFace

3. Monitor performance and adjust parameters as needed

## Getting Help

If you encounter issues:
1. Check the logs for error messages
2. Verify tinygrad is installed: `uv pip list | grep tinygrad`
3. Open an issue on the exo GitHub repository

## Comparison: MLX vs Tinygrad

| Feature | MLX | Tinygrad |
|---------|-----|----------|
| Platform | macOS, Linux x86_64 | All platforms |
| Hardware | GPU (Metal, CUDA) | CPU (GPU planned) |
| Performance | Very fast on Apple Silicon | Moderate on CPU |
| Memory | Efficient unified memory | Standard RAM |
| Models | Wide support | Llama models currently |
| Status | Production ready | Restored/experimental |
