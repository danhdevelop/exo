#!/bin/bash
# Quick verification script to check if GPU timeout fixes are properly configured

echo "=========================================="
echo "Verifying GPU Timeout Fix Configuration"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

errors=0
warnings=0

# Check constants.py
echo "Checking src/exo/worker/engines/mlx/constants.py..."
echo "-------------------------------------------"

if grep -q "KV_CACHE_BITS.*=.*4" src/exo/worker/engines/mlx/constants.py; then
    echo -e "${GREEN}✓${NC} KV_CACHE_BITS = 4 (correct for 128K support)"
else
    echo -e "${RED}✗${NC} KV_CACHE_BITS is not set to 4"
    errors=$((errors + 1))
fi

if grep -q "KV_BITS.*=.*4" src/exo/worker/engines/mlx/constants.py; then
    echo -e "${GREEN}✓${NC} KV_BITS = 4 (correct for 4-bit quantization)"
else
    echo -e "${RED}✗${NC} KV_BITS is not set to 4"
    errors=$((errors + 1))
fi

if grep -q "MAX_KV_SIZE.*=.*None" src/exo/worker/engines/mlx/constants.py; then
    echo -e "${GREEN}✓${NC} MAX_KV_SIZE = None (unlimited for 128K)"
else
    echo -e "${YELLOW}⚠${NC} MAX_KV_SIZE is not None (may limit context)"
    warnings=$((warnings + 1))
fi

echo ""

# Check generate.py
echo "Checking src/exo/worker/engines/mlx/generator/generate.py..."
echo "-------------------------------------------"

if grep -q "prefill_step_size=512" src/exo/worker/engines/mlx/generator/generate.py; then
    echo -e "${GREEN}✓${NC} prefill_step_size = 512 (correct for avoiding timeout)"
else
    echo -e "${RED}✗${NC} prefill_step_size is not set to 512"
    errors=$((errors + 1))
fi

# Count occurrences - should be 2 (warmup + generation)
prefill_count=$(grep -c "prefill_step_size=512" src/exo/worker/engines/mlx/generator/generate.py)
if [ "$prefill_count" -eq 2 ]; then
    echo -e "${GREEN}✓${NC} prefill_step_size set in both warmup and generation"
else
    echo -e "${YELLOW}⚠${NC} prefill_step_size found $prefill_count times (expected 2)"
    warnings=$((warnings + 1))
fi

# Check that we're using make_kv_cache without parameters
if grep -q "make_kv_cache(model=model)" src/exo/worker/engines/mlx/generator/generate.py; then
    echo -e "${GREEN}✓${NC} Using quantized KV cache (no max_kv_size parameter)"
else
    echo -e "${YELLOW}⚠${NC} KV cache creation might not be using quantization"
    warnings=$((warnings + 1))
fi

echo ""

# Check for helper scripts
echo "Checking helper scripts..."
echo "-------------------------------------------"

if [ -f "setup_metal_timeout.sh" ]; then
    echo -e "${GREEN}✓${NC} setup_metal_timeout.sh exists"
else
    echo -e "${YELLOW}⚠${NC} setup_metal_timeout.sh not found (optional)"
    warnings=$((warnings + 1))
fi

if [ -f "diagnose_metal_memory.py" ]; then
    echo -e "${GREEN}✓${NC} diagnose_metal_memory.py exists"
else
    echo -e "${YELLOW}⚠${NC} diagnose_metal_memory.py not found (optional)"
    warnings=$((warnings + 1))
fi

if [ -f "TROUBLESHOOTING_GPU_TIMEOUT.md" ]; then
    echo -e "${GREEN}✓${NC} TROUBLESHOOTING_GPU_TIMEOUT.md exists"
else
    echo -e "${YELLOW}⚠${NC} TROUBLESHOOTING_GPU_TIMEOUT.md not found (documentation)"
    warnings=$((warnings + 1))
fi

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="

if [ $errors -eq 0 ] && [ $warnings -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo ""
    echo "Configuration is correct for 128K token support with GPU timeout fix."
    echo ""
    echo "Next steps:"
    echo "1. (Optional) Run: python diagnose_metal_memory.py"
    echo "2. (Optional) Run: source setup_metal_timeout.sh"
    echo "3. Start your exo cluster and test with large contexts"
    exit_code=0
elif [ $errors -eq 0 ]; then
    echo -e "${YELLOW}⚠ Configuration mostly correct with $warnings warning(s)${NC}"
    echo ""
    echo "Configuration should work, but check warnings above."
    exit_code=0
else
    echo -e "${RED}✗ Found $errors error(s) and $warnings warning(s)${NC}"
    echo ""
    echo "Please fix the errors above before proceeding."
    echo "See TROUBLESHOOTING_GPU_TIMEOUT.md for details."
    exit_code=1
fi

echo ""
exit $exit_code
