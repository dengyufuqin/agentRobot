#!/bin/bash
#SBATCH --job-name=bench-mani
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/bench-mani-%j.log

# ============================================================
#  Benchmark: Octo × ManiSkill (PickCube, StackCube, PushCube)
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl

echo "============================================"
echo "  BENCHMARK: Octo × ManiSkill"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Start Octo server
echo "=== Starting Octo server ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/octo \
  $AGENTROBOT_ROOT/octo/.venv/bin/python3 $AGENTROBOT_ROOT/octo/policy_server.py \
  --port 18830 &
PID=$!

for i in $(seq 1 120); do
  curl -s http://localhost:18830/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

run_maniskill_eval() {
  local env_id=$1
  echo ""
  echo "=========================================="
  echo "  EVAL: Octo on ManiSkill/$env_id ($(date))"
  echo "=========================================="

  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/ManiSkill \
    $AGENTROBOT_ROOT/ManiSkill/.venv/bin/python3 -u $AGENTROBOT_ROOT/ManiSkill/scripts/run_eval.py \
    --env_id "$env_id" \
    --policy_server_addr localhost:18830 \
    --policy octo \
    --num_trials 10 \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results \
    --no_save_video

  echo "  → Octo on ManiSkill/$env_id: exit $?"
}

# Run ManiSkill evals
run_maniskill_eval "PickCube-v1"
run_maniskill_eval "StackCube-v1"
run_maniskill_eval "PushCube-v1"

# Cleanup
kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo ""
echo "============================================"
echo "  ManiSkill BENCHMARK COMPLETE"
echo "  $(date)"
echo "============================================"
