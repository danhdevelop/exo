#!/usr/bin/env python3
"""Diagnostic script to check what model weights you're loading"""

import json
import sys
from pathlib import Path
from safetensors import safe_open

def diagnose_model(model_path: str):
    """Diagnose model configuration from weights"""
    model_path = Path(model_path)

    print(f"\n=== Model Diagnostic ===")
    print(f"Path: {model_path}")
    print(f"Exists: {model_path.exists()}")

    if not model_path.exists():
        print("ERROR: Model path does not exist!")
        return

    # Check for config.json
    config_path = model_path / "config.json" if model_path.is_dir() else model_path.parent / "config.json"
    if config_path.exists():
        print(f"\n=== Config.json found ===")
        with open(config_path) as f:
            config = json.load(f)

        print(f"Model type: {config.get('model_type', 'unknown')}")
        print(f"Hidden size (dim): {config.get('hidden_size', 'N/A')}")
        print(f"Num attention heads (n_heads): {config.get('num_attention_heads', 'N/A')}")
        print(f"Num key-value heads (n_kv_heads): {config.get('num_key_value_heads', 'N/A')}")
        print(f"Num layers: {config.get('num_hidden_layers', 'N/A')}")
        print(f"Vocab size: {config.get('vocab_size', 'N/A')}")
        print(f"Intermediate size (hidden_dim): {config.get('intermediate_size', 'N/A')}")

        # Calculate expected wq shape
        dim = config.get('hidden_size')
        n_heads = config.get('num_attention_heads')
        if dim and n_heads:
            head_dim = dim // n_heads
            expected_wq = (dim, n_heads * head_dim)
            print(f"\n=== Expected wq.weight shape: {expected_wq} ===")

    # Check safetensors
    safetensors_files = []
    if model_path.is_dir():
        safetensors_files = list(model_path.glob("*.safetensors"))
        if (model_path / "model.safetensors.index.json").exists():
            print(f"\n=== Found sharded model ===")
            with open(model_path / "model.safetensors.index.json") as f:
                index = json.load(f)
                print(f"Weight map keys (first 10):")
                for i, key in enumerate(list(index.get('weight_map', {}).keys())[:10]):
                    print(f"  - {key}")
    else:
        if str(model_path).endswith('.safetensors'):
            safetensors_files = [model_path]

    if safetensors_files:
        print(f"\n=== Checking safetensors files ===")
        for sf_file in safetensors_files[:1]:  # Check first file only
            print(f"\nFile: {sf_file.name}")
            with safe_open(sf_file, framework="pt") as f:
                keys = list(f.keys())
                print(f"Number of tensors: {len(keys)}")
                print(f"\nFirst 20 tensor names and shapes:")
                for key in keys[:20]:
                    tensor = f.get_tensor(key)
                    print(f"  {key}: {tensor.shape}")

                # Find wq weight
                wq_keys = [k for k in keys if 'wq' in k or 'q_proj' in k]
                if wq_keys:
                    print(f"\n=== Found query projection layers ===")
                    for key in wq_keys[:3]:
                        tensor = f.get_tensor(key)
                        print(f"  {key}: {tensor.shape}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_model.py <model_path>")
        print("Example: python diagnose_model.py ~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B")
        sys.exit(1)

    diagnose_model(sys.argv[1])
