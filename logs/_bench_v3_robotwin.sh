#!/bin/bash
#SBATCH --job-name=bench-rtwin
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/bench-rtwin-%j.log

# ============================================================
#  Benchmark: Octo × RoboTwin (3 representative tasks)
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl

echo "============================================"
echo "  BENCHMARK: Octo × RoboTwin"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Start Octo server
echo "=== Starting Octo server ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/octo \
  $AGENTROBOT_ROOT/octo/.venv/bin/python3 $AGENTROBOT_ROOT/octo/policy_server.py \
  --port 18831 &
PID=$!

for i in $(seq 1 120); do
  curl -s http://localhost:18831/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

run_robotwin_eval() {
  local task=$1
  echo ""
  echo "=========================================="
  echo "  EVAL: Octo on RoboTwin/$task ($(date))"
  echo "=========================================="

  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/RoboTwin \
    $AGENTROBOT_ROOT/RoboTwin/.venv/bin/python3 -u $AGENTROBOT_ROOT/RoboTwin/script/run_eval_ws.py \
    --task_name "$task" \
    --policy_server_addr localhost:18831 \
    --num_trials 5 \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results \
    --no_save_video

  echo "  → Octo on RoboTwin/$task: exit $?"
}

# Run RoboTwin evals (representative tasks)
run_robotwin_eval "stack_blocks_two"
run_robotwin_eval "handover_block"
run_robotwin_eval "pick_diverse_bottles"

# Cleanup
kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo ""
echo "============================================"
echo "  RoboTwin BENCHMARK COMPLETE"
echo "  $(date)"
echo "============================================"
