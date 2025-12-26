# Running exo in CPU Mode

## Current Status

✅ **CPU Mode**: Working and stable
❌ **GPU Mode**: Not supported on Snapdragon X Elite (experimental)

## How to Run

### Default (CPU Mode - Recommended)
```bash
uv run python3 -m exo.main
```

The application will automatically:
- Default to CPU mode
- Skip GPU detection (prevents crashes)
- Run LLM inference on CPU cores

### Expected Behavior

**CPU Usage**: High CPU usage (50-100%) during inference is **normal and expected**
- This is because the model is computing on CPU
- NOT a bug - it's how CPU inference works
- Models are computationally intensive

## GPU Support Status

### What We Tried

1. **WebGPU/Vulkan** (via wgpu-native)
   - Tested versions: v22, v25, v27
   - Fixed multiple API compatibility issues
   - Result: Command buffer validation errors
   - Status: ❌ Not working

2. **OpenCL** (Adreno SDK)
   - Attempted: Qualcomm Adreno OpenCL ML SDK
   - Result: Android-only, incompatible with Linux
   - Status: ❌ Not available

3. **Manual GPU Override** (Not Recommended)
   ```bash
   # You can try GPU manually, but it will likely crash
   DEVICE=WEBGPU uv run python3 -m exo.main  # Will crash
   DEVICE=CL uv run python3 -m exo.main      # No drivers available
   ```

## GPU Support on Other Platforms

### Intel Hardware (Windows/Linux)

**Best Option**: OpenCL backend
```bash
# On Intel systems with OpenCL drivers
DEVICE=CL uv run python3 -m exo.main
```

**Check OpenCL availability**:
```bash
# Linux
sudo apt install clinfo
clinfo

# Windows
# Install Intel OpenCL drivers from Intel website
```

### Apple Silicon (macOS)

**MLX Engine** (native Apple Silicon support):
```bash
# Uses Metal GPU acceleration
uv run python3 -m exo.main
```
- MLX engine is automatically selected on macOS
- Full GPU acceleration via Metal
- Best performance on Apple Silicon

### NVIDIA GPUs

**CUDA backend**:
```bash
DEVICE=CUDA uv run python3 -m exo.main
```
- Requires NVIDIA CUDA drivers
- TinyGrad supports CUDA backend

## Performance Tips for CPU Mode

1. **Use Quantized Models**: 4-bit models use less memory and compute
2. **Limit Concurrent Requests**: Reduce parallel inference tasks
3. **Adjust Context Window**: Smaller context = less computation
4. **Close Other Apps**: Free up CPU resources

## Troubleshooting

### "Unsupported timed WaitAny features" Error

This happens if WebGPU tries to initialize. Solution:
- Don't set `DEVICE=WEBGPU`
- Use default CPU mode (no DEVICE variable)
- The app now skips GPU detection to prevent this

### High CPU Usage

This is **normal** for CPU-based LLM inference. Models are compute-intensive.

### Slow Inference

CPU inference is slower than GPU. This is expected. Options:
- Use a smaller/quantized model
- Use a system with GPU support (Intel with OpenCL, Apple Silicon, NVIDIA)
- Wait for Snapdragon X Elite GPU support to mature

## Technical Details

### Snapdragon X Elite GPU Status

**Hardware**:
- GPU: Adreno X1-85
- Vulkan: Detected and working
- WebGPU: Crashes during initialization

**Software Issues**:
- wgpu-native incompatibility with TinyGrad
- No Linux OpenCL drivers for Adreno
- Vulkan compute via WebGPU has validation errors

**Conclusion**: GPU acceleration on Snapdragon X Elite Linux is experimental and not yet functional.

## Future GPU Support

GPU support may improve when:
- Better Vulkan/GPU drivers for Snapdragon X Elite on Linux
- wgpu-native compatibility improves
- TinyGrad WebGPU backend matures
- Qualcomm releases Linux OpenCL drivers

## Questions?

- Check `WEBGPU_VULKAN_ISSUES.md` for detailed GPU debugging history
- CPU mode is stable and recommended for current use
- High CPU usage is expected and normal for CPU inference
