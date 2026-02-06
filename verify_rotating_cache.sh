#!/bin/bash
# Verify rotating KV cache is active on both Mac Minis

PASSWORD="fakerealpassword"

echo "=========================================="
echo "Verifying Rotating KV Cache Deployment"
echo "=========================================="
echo ""

echo "Checking mac-mini-1..."
echo "====================="
sshpass -p "$PASSWORD" ssh fakereal@mac-mini-1 "tail -100 /tmp/exo.log | grep 'rotating' || echo '❌ No rotating cache message found'"
echo ""

echo "Checking mac-mini-2..."
echo "====================="
sshpass -p "$PASSWORD" ssh fakereal01@mac-mini-2 "tail -100 /tmp/exo.log | grep 'rotating' || echo '❌ No rotating cache message found'"
echo ""

echo "Checking for GPU timeout errors..."
echo "==================================="
mini1_errors=$(sshpass -p "$PASSWORD" ssh fakereal@mac-mini-1 "tail -100 /tmp/exo.log | grep -c 'GPU Timeout' || echo 0")
mini2_errors=$(sshpass -p "$PASSWORD" ssh fakereal01@mac-mini-2 "tail -100 /tmp/exo.log | grep -c 'GPU Timeout' || echo 0")

if [ "$mini1_errors" -eq 0 ] && [ "$mini2_errors" -eq 0 ]; then
    echo "✅ No GPU timeout errors detected"
else
    echo "⚠️  GPU timeout errors found:"
    echo "   mac-mini-1: $mini1_errors errors"
    echo "   mac-mini-2: $mini2_errors errors"
fi
echo ""

echo "Memory usage (estimated from logs)..."
echo "======================================"
sshpass -p "$PASSWORD" ssh fakereal@mac-mini-1 "vm_stat | head -10" | grep -E "Pages (free|active|wired)" || echo "Could not get memory stats"
echo ""

echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Expected logs to see:"
echo "  ✓ 'Using rotating KV cache with max_kv_size=8000 with keep=4000'"
echo "  ✓ 'Detected 4-bit quantization from model ID'"
echo "  ✓ No GPU timeout errors"
echo ""
echo "If rotating cache message is missing:"
echo "  1. Wait 30-60 seconds and run this script again"
echo "  2. Check if exo is actually running: ps aux | grep exo"
echo "  3. Restart deployment: ./deploy_rotating_cache_fix.sh"
