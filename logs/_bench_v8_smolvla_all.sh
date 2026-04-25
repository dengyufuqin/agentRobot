#!/bin/bash
#SBATCH --job-name=eval-smolvla
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-smolvla-%j.log

# ============================================================
#  SmolVLA × LIBERO (4 suites) + ManiSkill (7 tasks)
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
PORT=18831

echo "============================================"
echo "  SmolVLA × LIBERO + ManiSkill (v8)"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Start SmolVLA server (uses lerobot policy_server.py)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/lerobot/src \
  $AGENTROBOT_ROOT/lerobot/.venv/bin/python3 $AGENTROBOT_ROOT/lerobot/policy_server.py \
  --checkpoint HuggingFaceVLA/smolvla_libero --port $PORT &
PID=$!

for i in $(seq 1 240); do
  curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

if ! curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
  echo "ERROR: SmolVLA server never became ready"
  kill $PID 2>/dev/null
  exit 1
fi

# === LIBERO Evaluations ===
echo ""
echo "=== LIBERO Evaluations ==="

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo ""
  echo "=========================================="
  echo "  EVAL: SmolVLA on LIBERO/$SUITE ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
    $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy smolvla --benchmark_name "$SUITE" \
    --num_trials 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
  echo "  → SmolVLA on LIBERO/$SUITE: exit $?"
done

# === ManiSkill Evaluations ===
echo ""
echo "=== ManiSkill Evaluations ==="

for TASK in PickCube-v1 StackCube-v1 PushCube-v1 PegInsertionSide-v1 TurnFaucet-v1 LiftPegUpright-v1 PlugCharger-v1; do
  echo ""
  echo "=========================================="
  echo "  EVAL: SmolVLA on ManiSkill/$TASK ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/ManiSkill \
    $AGENTROBOT_ROOT/ManiSkill/.venv/bin/python3 -u $AGENTROBOT_ROOT/ManiSkill/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy smolvla --env_id "$TASK" \
    --num_trials 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
  echo "  → SmolVLA on ManiSkill/$TASK: exit $?"
done

kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo ""
echo "============================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  $(date)"
echo "============================================"
