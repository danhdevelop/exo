#!/usr/bin/env python3
"""
Diagnostic script to check Metal GPU memory configuration on Mac M4.
This helps identify memory pressure issues causing kIOGPUCommandBufferCallbackErrorTimeout.
"""

import json
import mlx.core as mx

def main():
    print("=" * 60)
    print("Metal GPU Memory Diagnostics")
    print("=" * 60)

    if not mx.metal.is_available():
        print("❌ Metal is not available on this system")
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

    print("\n" + "=" * 60)
    print("Memory Configuration:")
    print("=" * 60)
    print(f"Max recommended working set: {max_rec / 1024**3:.2f} GB ({max_rec / 1024**2:.0f} MB)")
    print(f"Current cache limit:         {cache_limit / 1024**3:.2f} GB ({cache_limit / 1024**2:.0f} MB)")
    print(f"Current wired limit:         {wired_limit / 1024**3:.2f} GB ({wired_limit / 1024**2:.0f} MB)")

    # Estimate KV cache size for 8000 tokens
    print("\n" + "=" * 60)
    print("KV Cache Size Estimates (for 8000 token context):")
    print("=" * 60)

    # Typical model configurations
    configs = [
        {"name": "Llama 7B", "layers": 32, "hidden": 4096, "heads": 32},
        {"name": "Llama 13B", "layers": 40, "hidden": 5120, "heads": 40},
        {"name": "Llama 70B", "layers": 80, "hidden": 8192, "heads": 64},
    ]

    for config in configs:
        # KV cache size = 2 (K+V) * layers * seq_len * hidden_size * bytes_per_element
        # For fp16: 2 bytes, for 8-bit quantized: 1 byte
        tokens = 8000
        layers = config["layers"]
        hidden = config["hidden"]

        # FP16 size
        kv_fp16_bytes = 2 * layers * tokens * hidden * 2
        # 8-bit quantized size
        kv_8bit_bytes = 2 * layers * tokens * hidden * 1

        print(f"\n{config['name']}:")
        print(f"  FP16 KV cache:       {kv_fp16_bytes / 1024**2:.0f} MB ({kv_fp16_bytes / 1024**3:.2f} GB)")
        print(f"  8-bit quantized KV:  {kv_8bit_bytes / 1024**2:.0f} MB ({kv_8bit_bytes / 1024**3:.2f} GB)")

        # Check if it fits in recommended memory
        if kv_fp16_bytes > max_rec * 0.3:  # KV cache should be < 30% of available memory
            print(f"  ⚠️  WARNING: FP16 KV cache may cause memory pressure!")
        if kv_8bit_bytes > max_rec * 0.3:
            print(f"  ⚠️  WARNING: Even 8-bit KV cache may cause memory pressure!")

    print("\n" + "=" * 60)
    print("Recommendations:")
    print("=" * 60)
    print("1. Use quantized KV cache (8-bit or 4-bit) for large contexts")
    print("2. Use RotatingKVCache to limit max context size")
    print("3. Reduce prefill_step_size to process in smaller chunks")
    print("4. Increase Metal GPU timeout via environment variables")
    print("5. Monitor memory pressure during inference")

if __name__ == "__main__":
    main()
