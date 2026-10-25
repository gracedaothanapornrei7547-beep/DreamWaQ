#!/bin/bash
set -e

# ==============================================================================
# DreamWaQ TensorBoard Dashboard Launcher
# ==============================================================================

PORT=${1:-6006}
HOST=${2:-0.0.0.0}
LOGDIR="/home/rum/isaac/dreamwaq_lab/logs/rsl_rl/unitree"
TENSORBOARD_BIN="/home/rum/env_isaaclab_himloco/bin/tensorboard"

echo "=========================================================="
echo "  Launching TensorBoard Monitoring Dashboard"
echo "  Logdir : $LOGDIR"
echo "  Binding: http://$HOST:$PORT (Local: http://localhost:$PORT)"
echo "=========================================================="

exec $TENSORBOARD_BIN --logdir "$LOGDIR" --host "$HOST" --port "$PORT"

