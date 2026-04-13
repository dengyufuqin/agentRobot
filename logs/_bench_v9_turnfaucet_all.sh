#!/bin/bash
#SBATCH --job-name=eval-faucet
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-faucet-%j.log

# ============================================================
#  TurnFaucet-v1 reruns: pi0.5, Octo, OpenVLA-spatial
#  Assets pre-downloaded, should work now
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl

echo "============================================"
echo "  TurnFaucet-v1 multi-model rerun (v9)"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# --- pi0.5 ---
PORT=18840
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/lerobot/src \
  $AGENTROBOT_ROOT/lerobot/.venv/bin/python3 $AGENTROBOT_ROOT/lerobot/policy_server.py \
  --checkpoint lerobot/pi05_libero_finetuned_v044 --port $PORT &
PID=$!
for i in $(seq 1 240); do
  curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "pi0.5 server died!"; break; }; sleep 10
done

if curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
  echo "=========================================="
  echo "  EVAL: pi0.5 on ManiSkill/TurnFaucet-v1 ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/ManiSkill \
    $AGENTROBOT_ROOT/ManiSkill/.venv/bin/python3 -u $AGENTROBOT_ROOT/ManiSkill/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy pi05 --env_id "TurnFaucet-v1" \
    --num_trials 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
  echo "  → pi0.5 on ManiSkill/TurnFaucet-v1: exit $?"
fi
kill $PID 2>/dev/null; wait $PID 2>/dev/null

# --- Octo ---
PORT=18841
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/octo \
  $AGENTROBOT_ROOT/octo/.venv/bin/python3 $AGENTROBOT_ROOT/octo/policy_server.py \
  --port $PORT &
PID=$!
for i in $(seq 1 120); do
  curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "Octo server died!"; break; }; sleep 10
done

if curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
  echo "=========================================="
  echo "  EVAL: Octo on ManiSkill/TurnFaucet-v1 ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/ManiSkill \
    $AGENTROBOT_ROOT/ManiSkill/.venv/bin/python3 -u $AGENTROBOT_ROOT/ManiSkill/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy octo --env_id "TurnFaucet-v1" \
    --num_trials 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
  echo "  → Octo on ManiSkill/TurnFaucet-v1: exit $?"
fi
kill $PID 2>/dev/null; wait $PID 2>/dev/null

# --- OpenVLA-spatial ---
PORT=18842
cd $AGENTROBOT_ROOT/openvla
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --port $PORT &
PID=$!
for i in $(seq 1 120); do
  curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && break
  kill -0 $PID 2>/dev/null || { echo "OpenVLA server died!"; break; }; sleep 10
done

if curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
  echo "=========================================="
  echo "  EVAL: OpenVLA-OFT on ManiSkill/TurnFaucet-v1 ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/ManiSkill \
    $AGENTROBOT_ROOT/ManiSkill/.venv/bin/python3 -u $AGENTROBOT_ROOT/ManiSkill/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy openvla --env_id "TurnFaucet-v1" \
    --num_trials 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
  echo "  → OpenVLA-OFT on ManiSkill/TurnFaucet-v1: exit $?"
fi
kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo ""
echo "============================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  $(date)"
echo "============================================"
