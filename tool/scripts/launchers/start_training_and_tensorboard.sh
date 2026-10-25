#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

PYTHON_EXEC="/home/rum/env_isaaclab_himloco/bin/python"
TENSORBOARD_EXEC="/home/rum/env_isaaclab_himloco/bin/tensorboard"
LOG_DIR="$SCRIPT_DIR/logs/rsl_rl/unitree/unitree_go2_dreamwaq_rough"

mkdir -p "$LOG_DIR"

echo "1. Starting TensorBoard on port 6006..."
pkill -f "tensorboard.*6006" 2>/dev/null || true
nohup "$TENSORBOARD_EXEC" --logdir="$LOG_DIR" --port=6006 --bind_all > "$SCRIPT_DIR/tensorboard.log" 2>&1 &
TB_PID=$!
echo "   TensorBoard started with PID: $TB_PID (Logs: $SCRIPT_DIR/tensorboard.log)"

echo "2. Starting DreamWaQ 3000-iteration training in background..."
nohup ./run_train.sh 1024 3000 > "$SCRIPT_DIR/training.log" 2>&1 &
TRAIN_PID=$!
echo "   Training started with PID: $TRAIN_PID (Logs: $SCRIPT_DIR/training.log)"

echo "=========================================================="
echo "  DreamWaQ Training & TensorBoard Successfully Started!"
echo "  TensorBoard: http://localhost:6006"
echo "  Training PID: $TRAIN_PID"
echo "  TensorBoard PID: $TB_PID"
echo "=========================================================="
