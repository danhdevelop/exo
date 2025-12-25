#!/usr/bin/env python3
"""
Diagnostic script to check Metal GPU memory configuration on Mac M4.
This helps identify memory pressure issues causing kIOGPUCommandBufferCallbackErrorTimeout.

Updated for 4-bit quantized KV cache supporting 128K tokens.
"""

import json
import sys

try:
    import mlx.core as mx
except ImportError:
    print("❌ MLX is not installed. This script requires MLX.")
    print("Install with: pip install mlx")
    sys.exit(1)


def format_bytes(bytes_val):
    """Format bytes to human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} PB"


def main():
    print("=" * 70)
    print("Metal GPU Memory Diagnostics for 128K Token Support")
    print("=" * 70)

    if not mx.metal.is_available():
        print("❌ Metal is not available on this system")
        print("This script is designed for Mac with M-series chips.")
        return

    print("✓ Metal is available\n")

    # Get device info
    info = mx.metal.device_info()
    print("Metal Device Info:")
    print(json.dumps({k: v for k, v in info.items()}, indent=2))

    # Memory limits
    max_rec = info.get('max_recommended_working_set_size', 0)
    cache_limit = mx.metal.cache_limit()
    wired_limit = mx.metal.wired_limit()

    print("\n" + "=" * 70)
    print("Current Memory Configuration:")
    print("=" * 70)
    print(f"Max recommended working set: {format_bytes(max_rec)} ({max_rec / 1024**2:.0f} MB)")
    print(f"Current cache limit:         {format_bytes(cache_limit)} ({cache_limit / 1024**2:.0f} MB)")
    print(f"Current wired limit:         {format_bytes(wired_limit)} ({wired_limit / 1024**2:.0f} MB)")

    # Estimate KV cache size for various contexts and quantizations
    print("\n" + "=" * 70)
    print("KV Cache Memory Requirements:")
    print("=" * 70)

    # Model configurations
    configs = [
        {"name": "Llama 7B", "layers": 32, "hidden": 4096, "heads": 32},
        {"name": "Llama 13B", "layers": 40, "hidden": 5120, "heads": 40},
        {"name": "Llama 70B", "layers": 80, "hidden": 8192, "heads": 64},
    ]

    context_sizes = [8000, 32000, 65536, 131072]  # 8K, 32K, 64K, 128K

    for config in configs:
        print(f"\n{config['name']} ({config['layers']} layers, hidden={config['hidden']}):")
        print("-" * 70)

        layers = config["layers"]
        hidden = config["hidden"]

        for context in context_sizes:
            # KV cache size = 2 (K+V) * layers * seq_len * hidden_size * bytes_per_element
            kv_fp16 = 2 * layers * context * hidden * 2  # FP16: 2 bytes
            kv_8bit = 2 * layers * context * hidden * 1  # 8-bit: 1 byte
            kv_4bit = 2 * layers * context * hidden * 0.5  # 4-bit: 0.5 bytes

            context_str = f"{context // 1024}K" if context >= 1024 else str(context)

            print(f"\n  {context_str} tokens:")
            print(f"    FP16 (16-bit):     {format_bytes(kv_fp16):>15} - {kv_fp16 / 1024**3:>6.2f} GB")
            print(f"    8-bit quantized:   {format_bytes(kv_8bit):>15} - {kv_8bit / 1024**3:>6.2f} GB")
            print(f"    4-bit quantized:   {format_bytes(kv_4bit):>15} - {kv_4bit / 1024**3:>6.2f} GB  ← CURRENT")

            # Status indicators
            if kv_4bit < max_rec * 0.5:  # < 50% of available memory
                status = "✅ Comfortable fit"
            elif kv_4bit < max_rec * 0.8:  # < 80%
                status = "✅ Should work"
            elif kv_4bit < max_rec:  # < 100%
                status = "⚠️  Tight, may work"
            else:
                status = "❌ Too large, will likely fail"

            print(f"    Status (4-bit):    {status}")

    # Distributed inference estimates
    print("\n" + "=" * 70)
    print("Distributed Inference (2 nodes, pipeline parallelism):")
    print("=" * 70)
    print("Each node processes ~50% of layers, so KV cache is split:\n")

    config = configs[2]  # 70B model
    print(f"{config['name']} distributed across 2 nodes:")
    print("-" * 70)

    layers_per_node = config["layers"] // 2
    hidden = config["hidden"]

    for context in context_sizes:
        kv_4bit_per_node = 2 * layers_per_node * context * hidden * 0.5

        context_str = f"{context // 1024}K" if context >= 1024 else str(context)

        print(f"\n  {context_str} tokens:")
        print(f"    4-bit KV per node: {format_bytes(kv_4bit_per_node):>15} - {kv_4bit_per_node / 1024**3:>6.2f} GB")

        if kv_4bit_per_node < max_rec * 0.5:
            status = "✅ Comfortable fit on 16GB"
        elif kv_4bit_per_node < max_rec * 0.8:
            status = "✅ Should work on 16GB"
        elif kv_4bit_per_node < max_rec:
            status = "⚠️  Tight on 16GB"
        else:
            status = "❌ Need more RAM per node"

        print(f"    Status:            {status}")

    # Summary and recommendations
    print("\n" + "=" * 70)
    print("Configuration Verification:")
    print("=" * 70)

    # Check if constants are properly set
    try:
        from exo.worker.engines.mlx.constants import (
            KV_CACHE_BITS,
            KV_BITS,
            MAX_KV_SIZE,
        )

        print(f"KV_CACHE_BITS:  {KV_CACHE_BITS}")
        print(f"KV_BITS:        {KV_BITS}")
        print(f"MAX_KV_SIZE:    {MAX_KV_SIZE}")

        if KV_CACHE_BITS == 4 and KV_BITS == 4:
            print("\n✅ Configuration is correct for 128K token support")
        elif KV_CACHE_BITS == 8:
            print("\n⚠️  Using 8-bit quantization (max ~64K tokens recommended)")
        else:
            print("\n❌ Configuration issue: Should use 4-bit quantization for 128K support")

    except ImportError:
        print("\n⚠️  Could not verify exo configuration (import failed)")

    print("\n" + "=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print("1. ✅ Use 4-bit quantized KV cache (KV_CACHE_BITS=4, KV_BITS=4)")
    print("2. ✅ Use small prefill_step_size=512 to avoid GPU timeout")
    print("3. ✅ Set MAX_KV_SIZE=None for unlimited context up to 128K")
    print("4. ⚠️  Monitor memory pressure during inference")
    print("5. ⚠️  For 128K on 16GB M4: 7B models work, 13B+ may be tight")
    print("6. ✅ Optional: source setup_metal_timeout.sh for increased GPU timeout")

    print("\n" + "=" * 70)
    print("Memory Efficiency Comparison:")
    print("=" * 70)
    print(f"{'Quantization':<15} {'Memory Usage':<15} {'vs FP16':<15} {'Max Context (16GB)':<20}")
    print("-" * 70)
    print(f"{'FP16 (16-bit)':<15} {'1.0x':<15} {'baseline':<15} {'~32K tokens':<20}")
    print(f"{'8-bit':<15} {'0.5x':<15} {'50% savings':<15} {'~64K tokens':<20}")
    print(f"{'4-bit (current)':<15} {'0.25x':<15} {'75% savings':<15} {'~128K tokens':<20}")

    print("\n✓ Diagnostic complete\n")


if __name__ == "__main__":
    main()
