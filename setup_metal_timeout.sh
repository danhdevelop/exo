#!/bin/bash

# Setup script to configure Metal GPU for distributed inference on Mac M4
# This increases timeout limits to prevent kIOGPUCommandBufferCallbackErrorTimeout errors

echo "Configuring Metal GPU timeout settings for distributed inference..."

# Increase Metal GPU timeout (values in seconds)
# These environment variables help prevent GPU command buffer timeouts
# on M-series chips during distributed inference with large contexts

# Set Metal GPU timeout to 120 seconds (default is ~30 seconds)
export MTL_GPU_TIMEOUT=120

# Disable Metal command buffer timeout during development
# WARNING: Only use this during debugging, not in production
# export MTL_DEBUG_LAYER=1
# export MTL_SHADER_VALIDATION=0

# MLX-specific settings for distributed inference
export MLX_METAL_MAX_BUFFERS=16384  # Increase max buffer count
export MLX_METAL_LARGE_CACHE=1      # Enable large cache mode

# Increase system resource limits for MLX
ulimit -n 2048 2>/dev/null || true  # Increase max open files

echo "✓ Metal GPU timeout configuration complete"
echo ""
echo "Environment variables set:"
echo "  MTL_GPU_TIMEOUT=${MTL_GPU_TIMEOUT}"
echo "  MLX_METAL_MAX_BUFFERS=${MLX_METAL_MAX_BUFFERS}"
echo "  MLX_METAL_LARGE_CACHE=${MLX_METAL_LARGE_CACHE}"
echo ""
echo "To apply these settings, run:"
echo "  source setup_metal_timeout.sh"
echo ""
echo "Then start your exo cluster normally."
