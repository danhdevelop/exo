# Troubleshooting GPU Timeout Issues on Mac M4 Distributed Inference

## Problem Description

When running distributed inference on 2 Mac Mini M4 (16GB) machines with large contexts (~8000 tokens), the second node crashes with:

```
kIOGPUCommandBufferCallbackErrorTimeout (error code: 0x00000002)
```

The first node continues running, but the second node fails during inference.

## Root Cause Analysis

The `kIOGPUCommandBufferCallbackErrorTimeout` error occurs when GPU operations exceed the Metal GPU timeout limit (typically 30 seconds). This happens due to:

1. **Unbounded KV Cache Growth**: The code was creating unlimited KV caches for large contexts. For 8000 tokens, this causes:
   - Excessive memory allocation on 16GB M4
   - Memory pressure leading to slower GPU operations
   - GPU timeout when operations take > 30 seconds

2. **Large Prefill Batch Size**: `prefill_step_size=8192` processes almost the entire 8000-token context in one GPU batch, which can exceed timeout limits

3. **Distributed Pipeline Communication**: In pipeline parallelism:
   - Node 2 waits for Node 1 to process its layers
   - Node 2 receives intermediate results via network
   - Node 2 then processes the full batch through its layers
   - Total time can exceed GPU timeout

4. **Second Node More Vulnerable**: The second node experiences:
   - Network communication latency from first node
   - Same memory pressure as first node
   - Accumulated processing delays

## Fixes Implemented

### 1. Enable Rotating KV Cache (PRIMARY FIX)

**File**: `src/exo/worker/engines/mlx/generator/generate.py`

Changed from unlimited KV cache:
```python
caches = make_kv_cache(model=model)  # ❌ Unbounded
```

To size-limited rotating cache:
```python
caches = make_kv_cache(
    model=model,
    max_kv_size=MAX_KV_SIZE,      # 3200 tokens max
    keep=KEEP_KV_SIZE or 0,       # Keep 1600 tokens from start
)
```

**How it works**:
- Limits KV cache to 3200 tokens maximum
- Keeps first 1600 tokens (system prompt, context)
- Keeps last 1600 tokens (recent conversation)
- Discards middle tokens when context > 3200
- Prevents unbounded memory growth

### 2. Reduce Prefill Batch Size

Changed `prefill_step_size` from 8192 to 2048:
```python
prefill_step_size=2048  # Process in smaller chunks
```

**Benefits**:
- Processes prompt in 4 chunks instead of 1-2 chunks
- Each GPU operation completes faster
- Reduces risk of timeout
- Better memory management

### 3. Metal GPU Timeout Configuration

**File**: `setup_metal_timeout.sh`

Run before starting exo:
```bash
source setup_metal_timeout.sh
```

This sets:
- `MTL_GPU_TIMEOUT=120` - Increases Metal GPU timeout to 120 seconds
- `MLX_METAL_MAX_BUFFERS=16384` - More GPU buffers
- `MLX_METAL_LARGE_CACHE=1` - Enable large cache mode

### 4. Memory Diagnostics Tool

**File**: `diagnose_metal_memory.py`

Run to check memory configuration:
```bash
python diagnose_metal_memory.py
```

Shows:
- Metal device info
- Current memory limits
- KV cache size estimates for different models
- Warnings if configuration may cause issues

## Testing the Fixes

### Before Testing

1. Apply the code changes (already done in this branch)
2. Set up environment:
   ```bash
   cd /home/dn-nguyen/Workspace/exo_labs/exo/exo
   source setup_metal_timeout.sh
   ```

3. Check Metal configuration:
   ```bash
   python diagnose_metal_memory.py
   ```

### Test with Large Context

On both Mac Mini M4 machines:

1. **Start Node 1** (Master):
   ```bash
   source setup_metal_timeout.sh
   exo  # or your normal startup command
   ```

2. **Start Node 2**:
   ```bash
   source setup_metal_timeout.sh
   exo  # with appropriate distributed settings
   ```

3. **Send Large Context Test** (~8000 tokens):
   - Use a prompt with ~8000 tokens
   - Monitor both nodes for errors
   - Check that Node 2 doesn't crash

### Expected Behavior

✅ **With Fixes**:
- Both nodes process the request
- KV cache limited to 3200 tokens
- Prefill happens in 2048-token chunks
- No GPU timeout errors
- Slightly increased latency due to smaller batches (acceptable tradeoff)

❌ **Without Fixes**:
- Node 2 crashes with `kIOGPUCommandBufferCallbackErrorTimeout`
- Node 1 continues but inference fails

## Configuration Tuning

### If You Still See Timeouts

1. **Further reduce prefill_step_size**:
   Edit `src/exo/worker/engines/mlx/generator/generate.py`:
   ```python
   prefill_step = 1024  # Even smaller chunks
   ```

2. **Reduce MAX_KV_SIZE**:
   Edit `src/exo/worker/engines/mlx/constants.py`:
   ```python
   MAX_KV_SIZE: int | None = 2048  # Smaller cache
   KEEP_KV_SIZE: int | None = 1024
   ```

3. **Increase Metal timeout further**:
   Edit `setup_metal_timeout.sh`:
   ```bash
   export MTL_GPU_TIMEOUT=300  # 5 minutes
   ```

### If Quality Degrades

If responses are less coherent (due to truncated KV cache):

1. **Increase MAX_KV_SIZE** (if you have memory):
   ```python
   MAX_KV_SIZE: int | None = 4096  # Larger cache
   KEEP_KV_SIZE: int | None = 2048
   ```

2. **Use quantized KV cache** (already enabled):
   - `KV_CACHE_BITS=8` uses 8-bit quantization
   - Saves 50% memory vs FP16
   - Minimal quality impact

### Model-Specific Settings

Large models (70B+) on 16GB M4:
```python
MAX_KV_SIZE: int | None = 2048  # Smaller for large models
prefill_step = 1024              # Smaller batches
```

Small models (7B-13B) on 16GB M4:
```python
MAX_KV_SIZE: int | None = 4096  # Can be larger
prefill_step = 2048              # Current default
```

## Monitoring

### Watch for These Indicators

1. **Memory Pressure**:
   ```bash
   # On Mac, monitor memory
   memory_pressure
   ```

2. **GPU Usage**:
   Check Activity Monitor > GPU tab for high memory usage

3. **Logs**:
   Look for these log messages:
   - `"Using rotating KV cache with max_kv_size=3200"` ✅
   - `"Using default KV cache"` ❌ (unbounded)

## Understanding the Tradeoffs

| Configuration | Memory Usage | Speed | Quality | Timeout Risk |
|--------------|--------------|-------|---------|--------------|
| Unlimited KV | Very High | Fast | Best | ❌ High |
| KV=4096 | High | Fast | Good | ⚠️ Medium |
| KV=3200 (default) | Medium | Medium | Good | ✅ Low |
| KV=2048 | Low | Medium | Fair | ✅ Very Low |

## Thunderbolt Connection

The issue is **NOT a Thunderbolt bottleneck**. The network communication is fast enough. The problem is:
- GPU processing time exceeding timeout
- Memory allocation and management
- Not network bandwidth

Thunderbolt 4/5 provides:
- 40-80 Gbps bandwidth
- ~5 GB/s actual throughput
- Sufficient for tensor communication

The timeout occurs during local GPU computation, not during network transfer.

## Additional Resources

- **Apple Metal Documentation**: https://developer.apple.com/metal/
- **MLX Documentation**: https://ml-explore.github.io/mlx/
- **MLX-LM KV Cache**: Check mlx-lm docs for RotatingKVCache details

## Summary

The primary fix is **enabling RotatingKVCache with size limits**. This prevents unbounded memory growth that causes GPU timeouts. Combined with smaller prefill batches and increased Metal timeout, the distributed inference should work reliably on 16GB M4 Macs even with large contexts.

---

**Files Modified**:
- `src/exo/worker/engines/mlx/generator/generate.py` - Main fix
- `src/exo/worker/engines/mlx/constants.py` - Already had correct values
- `setup_metal_timeout.sh` - New helper script
- `diagnose_metal_memory.py` - New diagnostic tool
