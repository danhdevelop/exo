#!/bin/bash
# Deploy rotating KV cache fix to both Mac Minis

set -e  # Exit on error

PASSWORD="fakerealpassword"

RSYNC_EXCLUDES=(
  --exclude '__pycache__/'
  --exclude '*.pyc'
  --exclude '*.pyo'
  --exclude '.pytest_cache/'
  --exclude '.mypy_cache/'
  --exclude '.ruff_cache/'
  --exclude 'node_modules/'
  --exclude 'venv/'
  --exclude '.venv/'
  --exclude '*.egg-info/'
  --exclude 'dist/'
  --exclude 'build/'
  --exclude '.git/'
  --exclude '.uv/'
  --exclude 'dashboard/build/'
)

echo "=========================================="
echo "Deploying Rotating KV Cache Fix"
echo "=========================================="
echo ""

# Step 1: Stop exo on both nodes
echo "Step 1: Stopping exo on both nodes..."
sshpass -p "$PASSWORD" ssh fakereal@mac-mini-1 "pkill -f 'uv run exo' || echo 'No exo running on mac-mini-1'" &
sshpass -p "$PASSWORD" ssh fakereal01@mac-mini-2 "pkill -f 'uv run exo' || echo 'No exo running on mac-mini-2'" &
wait
echo "✓ Stopped exo on both nodes"
echo ""

# Step 2: Sync files to both nodes
echo "Step 2: Syncing files to both nodes..."
sshpass -p "$PASSWORD" rsync -av "${RSYNC_EXCLUDES[@]}" \
  src/exo/worker/engines/mlx/cache.py \
  src/exo/worker/engines/mlx/constants.py \
  src/exo/worker/engines/mlx/generator/generate.py \
  fakereal@mac-mini-1:/Users/fakereal/exo_cline/exo/src/exo/worker/engines/mlx/ &

sshpass -p "$PASSWORD" rsync -av "${RSYNC_EXCLUDES[@]}" \
  src/exo/worker/engines/mlx/cache.py \
  src/exo/worker/engines/mlx/constants.py \
  src/exo/worker/engines/mlx/generator/generate.py \
  fakereal01@mac-mini-2:/Users/fakereal01/exo_cline/exo/src/exo/worker/engines/mlx/ &
wait
echo "✓ Files synced to both nodes"
echo ""

# Step 3: Start exo on both nodes
echo "Step 3: Starting exo on both nodes..."
echo "Starting mac-mini-1..."
sshpass -p "$PASSWORD" ssh fakereal@mac-mini-1 "cd /Users/fakereal/exo_cline/exo && nohup uv run exo > /tmp/exo.log 2>&1 &" &

echo "Starting mac-mini-2..."
sshpass -p "$PASSWORD" ssh fakereal01@mac-mini-2 "cd /Users/fakereal01/exo_cline/exo && nohup uv run exo > /tmp/exo.log 2>&1 &" &
wait
sleep 3
echo "✓ Started exo on both nodes"
echo ""

# Step 4: Check logs for confirmation
echo "Step 4: Checking logs for 'Using rotating KV cache' message..."
echo ""
echo "=== mac-mini-1 logs ==="
sshpass -p "$PASSWORD" ssh fakereal@mac-mini-1 "tail -20 /tmp/exo.log | grep -A2 -B2 'rotating' || echo 'No rotating cache message yet (may take a moment)'"
echo ""
echo "=== mac-mini-2 logs ==="
sshpass -p "$PASSWORD" ssh fakereal01@mac-mini-2 "tail -20 /tmp/exo.log | grep -A2 -B2 'rotating' || echo 'No rotating cache message yet (may take a moment)'"
echo ""

echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Wait ~30 seconds for nodes to fully start"
echo "2. Check logs on both nodes for 'Using rotating KV cache with max_kv_size=8000'"
echo "3. Test with your 33K token request"
echo ""
echo "To view logs:"
echo "  mac-mini-1: ssh fakereal@mac-mini-1 'tail -f /tmp/exo.log'"
echo "  mac-mini-2: ssh fakereal01@mac-mini-2 'tail -f /tmp/exo.log'"
echo ""
echo "Expected memory usage: ~8.4GB per node (was 13.3GB)"
echo "Should eliminate GPU timeout errors! ✅"
