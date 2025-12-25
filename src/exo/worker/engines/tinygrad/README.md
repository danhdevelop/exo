# Tinygrad Inference Engine

This directory contains the tinygrad inference engine implementation for exo.

## Overview

The tinygrad engine has been restored from the project's git history and adapted to work with the current exo architecture. It provides an alternative to the MLX engine for running inference.

## Structure

```
tinygrad/
├── __init__.py
├── README.md (this file)
├── constants.py          # Configuration constants
├── utils_tinygrad.py     # Initialization and utilities
├── tinygrad_helpers.py   # Helper functions for weight loading
├── stateful_model.py     # KV cache and model state management
├── generator/
│   ├── __init__.py
│   └── generate.py       # Text generation functions
└── models/
    ├── __init__.py
    └── llama.py          # Llama model implementation
```

## Installation

### On Linux ARM64 or platforms where MLX is not available:

The default installation now includes tinygrad as a required dependency:

```bash
# Clone and install
git clone https://github.com/exo-explore/exo
cd exo/dashboard && npm install && npm run build && cd ..
uv run exo
```

Tinygrad will automatically be used if MLX is not available.

### To explicitly use tinygrad (on any platform):

```bash
export INFERENCE_ENGINE=tinygrad
uv run exo
```

Or run directly with the environment variable:

```bash
INFERENCE_ENGINE=tinygrad uv run exo
```

### To use MLX (on macOS or Linux x86_64):

Install the optional MLX dependencies:

```bash
uv pip install -e ".[mlx]"
# Then run normally (MLX is the default)
uv run exo
```

## Configuration

The tinygrad engine uses the following environment variables for configuration:

- `INFERENCE_ENGINE`: Set to `tinygrad` to use this engine (default: `mlx`)
- `TINYGRAD_DEVICE` or `DEVICE`: Manually specify the device backend (e.g., `QCOM`, `WEBGPU`, `CL`, `CUDA`, `CPU`)
- `TEMPERATURE`: Sampling temperature (default: 0.85)
- `TOP_K`: Top-k sampling parameter (default: 25)
- `TOP_P`: Top-p (nucleus) sampling parameter (default: 0.9)
- `MAX_TOKENS`: Maximum tokens to generate (default: 8192)

### GPU and Vulkan Support

The tinygrad engine now automatically detects and uses the best available GPU backend. Device selection priority:

1. **QCOM** (Qualcomm) - For Snapdragon X Elite and other Qualcomm processors (uses Vulkan)
2. **WEBGPU** - Vulkan-based backend with wide hardware compatibility
3. **CL** (OpenCL) - May use Vulkan ICD on some systems
4. **METAL** - For Apple Silicon
5. **CUDA** - For NVIDIA GPUs
6. **HIP/AMD** - For AMD GPUs
7. **CPU** - Fallback when no GPU is available

This enables support for systems like Ubuntu on Snapdragon X Elite and other ARM-based platforms with Vulkan support.

To manually select a device:
```bash
TINYGRAD_DEVICE=QCOM uv run exo  # Force Qualcomm backend
TINYGRAD_DEVICE=WEBGPU uv run exo  # Force WebGPU/Vulkan backend
TINYGRAD_DEVICE=CPU uv run exo  # Force CPU backend
```

## Supported Models

The tinygrad engine currently supports Llama-based models with the following sizes:
- 1B
- 3B
- 8B
- 70B

## Dependencies

The tinygrad engine requires:
- `tinygrad` (installed via requirements)
- `transformers` (for tokenizers)
- `numpy`

## Implementation Notes

1. **Thread Safety**: Tinygrad operations run on a singleton thread pool executor to ensure thread safety
2. **Model Sharding**: Supports layer-based sharding for distributed inference
3. **KV Caching**: Implements key-value caching for efficient autoregressive generation
4. **Tokenization**: Uses HuggingFace transformers for tokenization

## Differences from MLX Engine

- Tinygrad supports multiple backends (QCOM/Vulkan, WEBGPU, OpenCL, CUDA, HIP, METAL, CPU)
- MLX is limited to Apple Silicon GPUs
- Tinygrad has wider hardware compatibility including Snapdragon X Elite and other ARM platforms
- Different memory management strategies
- Simpler distributed inference setup
- May have different performance characteristics depending on the backend

## Development Status

This is a restored implementation adapted to the current architecture. Some features may need additional testing and refinement:

- [ ] Distributed inference across multiple devices
- [ ] Performance optimization
- [ ] Extended model support beyond Llama
- [ ] Advanced sampling strategies
- [ ] Streaming optimizations

## Troubleshooting

If you encounter issues:

1. **Import Errors**: Ensure tinygrad is installed: `pip install tinygrad`
2. **Model Loading Failures**: Check model paths and ensure models are downloaded
3. **Memory Issues**: Adjust model size or enable sharding
4. **Tokenizer Issues**: Verify transformers library is installed

## Contributing

When contributing to the tinygrad engine:

1. Maintain compatibility with the current runner architecture
2. Follow the same patterns as the MLX engine for consistency
3. Add tests for new features
4. Update this README with any changes

## References

- [Tinygrad Repository](https://github.com/tinygrad/tinygrad)
- [Original exo tinygrad implementation](https://github.com/exo-explore/exo) (git history before v1)
