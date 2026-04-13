#!/bin/bash
#SBATCH --job-name=eval-octo
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-octo-%j.log

# ============================================================
#  Octo × LIBERO (all 4 suites) + ManiSkill + RoboTwin
#  With ptxas fix for JAX compilation
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl

# Fix: Add ptxas to PATH for JAX PTX compilation
PTXAS_DIR=$AGENTROBOT_ROOT/octo/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin
export PATH=$PTXAS_DIR:$PATH

echo "============================================"
echo "  Octo × LIBERO + ManiSkill + RoboTwin"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo "ptxas: $(which ptxas 2>/dev/null || echo 'NOT FOUND')"
echo ""

# Start Octo server
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/octo \
  $AGENTROBOT_ROOT/octo/.venv/bin/python3 $AGENTROBOT_ROOT/octo/policy_server.py \
  --port 18826 &
PID=$!

for i in $(seq 1 180); do
  curl -s http://localhost:18826/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

if ! curl -s http://localhost:18826/healthz >/dev/null 2>&1; then
  echo "ERROR: Octo server never became ready"
  kill $PID 2>/dev/null
  exit 1
fi

run_libero() {
  local suite=$1
  echo ""
  echo "=========================================="
  echo "  EVAL: Octo on LIBERO/$suite ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
    $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
    --policy_server_addr localhost:18826 --policy octo --task_suite_name "$suite" \
    --num_trials_per_task 5 --arm_controller cartesian_pose \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
  echo "  → Octo on LIBERO/$suite: exit $?"
}

run_maniskill() {
  local env_id=$1
  echo ""
  echo "=========================================="
  echo "  EVAL: Octo on ManiSkill/$env_id ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/ManiSkill \
    $AGENTROBOT_ROOT/ManiSkill/.venv/bin/python3 -u $AGENTROBOT_ROOT/ManiSkill/scripts/run_eval.py \
    --env_id "$env_id" --policy_server_addr localhost:18826 --policy octo \
    --num_trials 10 --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
  echo "  → Octo on ManiSkill/$env_id: exit $?"
}

run_robotwin() {
  local task=$1
  echo ""
  echo "=========================================="
  echo "  EVAL: Octo on RoboTwin/$task ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/RoboTwin \
    $AGENTROBOT_ROOT/RoboTwin/.venv/bin/python3 -u $AGENTROBOT_ROOT/RoboTwin/script/run_eval_ws.py \
    --task_name "$task" --policy_server_addr localhost:18826 \
    --num_trials 5 --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
  echo "  → Octo on RoboTwin/$task: exit $?"
}

echo ""
echo "=== LIBERO Evaluations ==="
run_libero "libero_spatial"
run_libero "libero_object"
run_libero "libero_goal"
run_libero "libero_10"

echo ""
echo "=== ManiSkill Evaluations ==="
run_maniskill "PickCube-v1"
run_maniskill "StackCube-v1"
run_maniskill "PushCube-v1"

echo ""
echo "=== RoboTwin Evaluations ==="
run_robotwin "stack_blocks_two"
run_robotwin "handover_block"
run_robotwin "pick_diverse_bottles"

# Cleanup
kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo ""
echo "============================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  $(date)"
echo "============================================"
