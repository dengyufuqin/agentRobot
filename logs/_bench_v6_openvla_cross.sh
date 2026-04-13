#!/bin/bash
#SBATCH --job-name=eval-ovla-x
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-ovla-x-%j.log

# ============================================================
#  OpenVLA-OFT × ManiSkill (cross-benchmark generality)
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
PORT=18800

echo "============================================"
echo "  OpenVLA-OFT × ManiSkill"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Start OpenVLA server
cd $AGENTROBOT_ROOT/openvla
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
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

echo ""
echo "=== ManiSkill Evaluations ==="

for TASK in PickCube-v1 StackCube-v1 PushCube-v1; do
  echo ""
  echo "=========================================="
  echo "  EVAL: OpenVLA-OFT on ManiSkill/$TASK ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/ManiSkill \
    $AGENTROBOT_ROOT/ManiSkill/.venv/bin/python3 -u $AGENTROBOT_ROOT/ManiSkill/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy openvla --env_id "$TASK" \
    --num_trials 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
  echo "  → OpenVLA-OFT on ManiSkill/$TASK: exit $?"
done

kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo ""
echo "============================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  $(date)"
echo "============================================"
