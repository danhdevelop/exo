#!/usr/bin/env python3
"""Check quantized weight shapes"""

from tinygrad.nn.state import safe_load
import sys

model_path = sys.argv[1] if len(sys.argv) > 1 else "~/.exo/models/mlx-community--Qwen3-0.6B-4bit/model.safetensors"

print(f"Loading {model_path}")
weights = safe_load(model_path)

print("\n=== Checking quantized layer 0 attention weights ===")
for key in sorted(weights.keys()):
    if "layers.0.self_attn.q_proj" in key or "layers.0.attention.wq" in key:
        print(f"{key}: {weights[key].shape} dtype={weights[key].dtype}")

print("\n=== All layer 0 keys ===")
for key in sorted(weights.keys()):
    if "layers.0" in key:
        print(f"{key}: {weights[key].shape} dtype={weights[key].dtype}")
