#!/bin/bash
#SBATCH --job-name=eval-ovla-u2
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-ovla-u2-%j.log

# ============================================================
#  OpenVLA-OFT (unified) × LIBERO (4 suites) + ManiSkill TurnFaucet
#  Fixed: --task_suite_name, --num_trials_per_task
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
PORT=18833

echo "============================================"
echo "  OpenVLA-OFT (unified) × LIBERO + ManiSkill (v9)"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Start OpenVLA server with unified checkpoint
cd $AGENTROBOT_ROOT/openvla
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 \
  --port $PORT &
PID=$!

for i in $(seq 1 120); do
  curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

if ! curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
  echo "ERROR: OpenVLA server never became ready"
  kill $PID 2>/dev/null
  exit 1
fi

# === LIBERO Evaluations ===
echo ""
echo "=== LIBERO Evaluations ==="

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo ""
  echo "=========================================="
  echo "  EVAL: OpenVLA-unified on LIBERO/$SUITE ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
    $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy openvla --task_suite_name "$SUITE" \
    --num_trials_per_task 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
  echo "  → OpenVLA-unified on LIBERO/$SUITE: exit $?"
done

# === ManiSkill TurnFaucet rerun ===
echo ""
echo "=== ManiSkill TurnFaucet Rerun ==="
echo ""
echo "=========================================="
echo "  EVAL: OpenVLA-unified on ManiSkill/TurnFaucet-v1 ($(date))"
echo "=========================================="
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/ManiSkill \
  $AGENTROBOT_ROOT/ManiSkill/.venv/bin/python3 -u $AGENTROBOT_ROOT/ManiSkill/scripts/run_eval.py \
  --policy_server_addr localhost:$PORT --policy openvla --env_id "TurnFaucet-v1" \
  --num_trials 5 --no_save_video \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
echo "  → OpenVLA-unified on ManiSkill/TurnFaucet-v1: exit $?"

kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo ""
echo "============================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  $(date)"
echo "============================================"
