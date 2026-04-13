#!/bin/bash
#SBATCH --job-name=eval-octo-rt
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-octo-rt-%j.log

# ============================================================
#  Octo × RoboTwin (v6 — curobo installed)
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
PORT=18826

echo "============================================"
echo "  Octo × RoboTwin (v6 — curobo installed)"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ptxas fix for JAX on H100
export PATH=$AGENTROBOT_ROOT/octo/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH

# Start Octo server
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src \
  $AGENTROBOT_ROOT/octo/.venv/bin/python3 $AGENTROBOT_ROOT/octo/policy_server.py \
  --port $PORT &
PID=$!

for i in $(seq 1 60); do
  curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

if ! curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
  echo "ERROR: Octo server never became ready"
  kill $PID 2>/dev/null
  exit 1
fi

echo ""
echo "=== RoboTwin Evaluations ==="

for TASK in stack_blocks_two handover_block pick_diverse_bottles; do
  echo ""
  echo "=========================================="
  echo "  EVAL: Octo on RoboTwin/$TASK ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src \
    $AGENTROBOT_ROOT/RoboTwin/.venv/bin/python3 -u $AGENTROBOT_ROOT/RoboTwin/script/run_eval_ws.py \
    --policy_server_addr localhost:$PORT --policy octo --task_name "$TASK" \
    --num_trials 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
  echo "  → Octo on RoboTwin/$TASK: exit $?"
done

kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo ""
echo "============================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  $(date)"
echo "============================================"
