# Setting Up GPU/Vulkan Support for Tinygrad on Snapdragon X Elite

This guide helps you set up GPU acceleration for the tinygrad engine on Ubuntu with Snapdragon X Elite.

## Step 1: Check Current GPU Support

Run this diagnostic to see what's available on your system:

```bash
# Check for Vulkan
vulkaninfo --summary 2>/dev/null || echo "❌ Vulkan tools not installed"

# Check for OpenCL
clinfo 2>/dev/null || echo "❌ OpenCL tools not installed"

# Check GPU devices
ls -la /dev/dri/ 2>/dev/null || echo "❌ No DRI devices found"
ls -la /dev/kgsl* 2>/dev/null || echo "❌ No KGSL/Adreno devices found"

# Check if GPU is recognized
lspci | grep -i vga
lspci | grep -i display
```

## Step 2: Install GPU Drivers and Runtimes

### For Snapdragon X Elite (Qualcomm Adreno GPU)

The Snapdragon X Elite has an integrated Adreno GPU that supports Vulkan.

#### Install Vulkan Runtime

```bash
# Install Vulkan loader and tools
sudo apt update
sudo apt install -y \
    vulkan-tools \
    vulkan-validationlayers \
    mesa-vulkan-drivers \
    libvulkan1

# For Qualcomm/Adreno specific drivers (if available)
sudo apt install -y mesa-vulkan-drivers:arm64
```

#### Install OpenCL Runtime (Alternative)

```bash
# Install OpenCL runtime
sudo apt install -y \
    ocl-icd-libopencl1 \
    opencl-headers \
    clinfo \
    mesa-opencl-icd
```

#### Install Mesa Graphics Stack (Important for ARM)

```bash
# Install Mesa for ARM GPU support
sudo apt install -y \
    mesa-utils \
    libegl1-mesa \
    libgles2-mesa \
    libgbm1

# Install firmware (may be needed for some devices)
sudo apt install -y linux-firmware
```

## Step 3: Verify Installation

After installation, verify that GPU support is working:

```bash
# Test Vulkan
vulkaninfo --summary

# Should show your Adreno GPU if successful
# Look for lines like:
#   GPU id       : 0 (Qualcomm Adreno...)
#   GPU name     : Qualcomm Adreno...

# Test OpenCL
clinfo

# Check tinygrad device detection
cd /home/dn-nguyen/Workspace/exo_labs/exo/exo
uv run python3 src/exo/worker/engines/tinygrad/test_device_detection.py
```

## Step 4: Configure Tinygrad

After installing GPU drivers, tinygrad should automatically detect available backends.

### Test with specific backends:

```bash
# Try WEBGPU (Vulkan-based)
TINYGRAD_DEVICE=WEBGPU uv run python3 -c "from tinygrad import Device; print('Using:', Device.DEFAULT)"

# Try OpenCL
TINYGRAD_DEVICE=CL uv run python3 -c "from tinygrad import Device; print('Using:', Device.DEFAULT)"

# Try QCOM (Qualcomm-specific)
TINYGRAD_DEVICE=QCOM uv run python3 -c "from tinygrad import Device; print('Using:', Device.DEFAULT)"
```

## Step 5: Run Exo with GPU

Once GPU support is confirmed:

```bash
# Let tinygrad auto-detect the best device
INFERENCE_ENGINE=tinygrad uv run exo

# Or force a specific device
TINYGRAD_DEVICE=WEBGPU INFERENCE_ENGINE=tinygrad uv run exo
TINYGRAD_DEVICE=CL INFERENCE_ENGINE=tinygrad uv run exo
TINYGRAD_DEVICE=QCOM INFERENCE_ENGINE=tinygrad uv run exo
```

## Troubleshooting

### Issue: "No GPU devices found"

**Solution:** Your GPU drivers may not be properly installed or loaded.

```bash
# Check if GPU is detected by kernel
dmesg | grep -i adreno
dmesg | grep -i gpu
dmesg | grep -i drm

# Check loaded modules
lsmod | grep -i drm
lsmod | grep -i gpu
```

### Issue: "Vulkan not available"

**Solution:** Install mesa-vulkan-drivers and verify Vulkan loader:

```bash
sudo apt install -y mesa-vulkan-drivers libvulkan1
ldconfig -p | grep vulkan
```

### Issue: Tinygrad still shows only CPU

**Possible causes:**
1. GPU drivers not installed correctly
2. Tinygrad's backend dependencies not available
3. Permissions issue accessing GPU

**Debug steps:**
```bash
# Check GPU permissions
ls -la /dev/dri/
# You should see renderD* nodes, and your user should have access

# Add yourself to render/video groups if needed
sudo usermod -a -G render,video $USER
# Log out and back in for this to take effect

# Check tinygrad available devices with verbose output
DEBUG=1 uv run python3 -c "from tinygrad.device import Device; print(list(Device.get_available_devices()))"
```

### Issue: "GPU works but performance is slow"

The first run may be slow due to shader compilation. Subsequent runs should be faster.

## Platform-Specific Notes

### Snapdragon X Elite on Ubuntu

- The Snapdragon X Elite uses Qualcomm Adreno GPU
- Vulkan is the recommended API (via WEBGPU or QCOM backends)
- Mesa provides open-source drivers for Adreno GPUs
- Proprietary Qualcomm drivers may offer better performance but are rarely available on Linux

### Expected Performance

- **CPU only**: Slowest, but works everywhere
- **OpenCL (CL)**: Moderate improvement
- **Vulkan (WEBGPU/QCOM)**: Best performance for Snapdragon X Elite

## Need Help?

Please share the output of:

```bash
# System info
uname -a
lspci | grep -i vga

# GPU detection
vulkaninfo --summary
clinfo

# Tinygrad detection
uv run python3 src/exo/worker/engines/tinygrad/test_device_detection.py

# Exo log output (first 100 lines)
INFERENCE_ENGINE=tinygrad uv run exo 2>&1 | head -100
```

This will help diagnose any remaining issues.
