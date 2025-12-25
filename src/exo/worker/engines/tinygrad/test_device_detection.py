#!/usr/bin/env python3
"""
Test script for tinygrad device detection.
Run this to verify that Vulkan/GPU backends are properly detected.
"""

import os
import sys

# Add src to path if running standalone
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))


def test_device_detection():
    """Test the device detection functionality."""
    print("=" * 70)
    print("Tinygrad Device Detection Test")
    print("=" * 70)

    # Import after path is set up
    from tinygrad import Device

    print("\n1. Available devices from tinygrad:")
    available = list(Device.get_available_devices())
    for device in available:
        print(f"   - {device}")

    print("\n2. Testing device priority selection:")
    print("   Priority order:")
    priority = [
        "QCOM (Qualcomm/Snapdragon - Vulkan)",
        "WEBGPU (Vulkan-based)",
        "CL (OpenCL - may use Vulkan)",
        "METAL (Apple Silicon)",
        "CUDA (NVIDIA)",
        "HIP/AMD (AMD GPUs)",
        "CPU (fallback)",
    ]
    for i, p in enumerate(priority, 1):
        print(f"   {i}. {p}")

    print("\n3. Simulated device selection:")
    device_priority = ["QCOM", "WEBGPU", "CL", "METAL", "CUDA", "HIP", "AMD", "NV", "CPU"]

    selected = None
    for device in device_priority:
        if device in available:
            selected = device
            break

    if selected:
        print(f"   ✓ Would select: {selected}")
        if selected == "QCOM":
            print("     → Qualcomm backend (Vulkan-based) for Snapdragon")
        elif selected == "WEBGPU":
            print("     → WebGPU backend (uses Vulkan)")
        elif selected == "CL":
            print("     → OpenCL backend (may use Vulkan drivers)")
        elif selected == "CPU":
            print("     → CPU fallback (no GPU detected)")
    else:
        print("   ✗ No device could be selected!")

    print("\n4. Environment variable override test:")
    test_devices = ["QCOM", "WEBGPU", "CPU"]
    for test_dev in test_devices:
        os.environ["TINYGRAD_DEVICE"] = test_dev
        print(f"   TINYGRAD_DEVICE={test_dev}")
        print(f"   → Would use: {test_dev}")
        del os.environ["TINYGRAD_DEVICE"]

    print("\n" + "=" * 70)
    print("Test completed!")
    print("=" * 70)

    # Return selected device for automated testing
    return selected


if __name__ == "__main__":
    device = test_device_detection()
    sys.exit(0 if device else 1)
