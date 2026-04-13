#!/bin/bash
#SBATCH --job-name=eval-pi05v2
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-pi05v2-%j.log

# ============================================================
#  pi0.5 × LIBERO spatial (with preprocessor fix + warmup)
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl

echo "============================================"
echo "  pi0.5 × LIBERO-Spatial (v2 — preprocessor fix)"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Start pi0.5 server
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/lerobot/src \
  $AGENTROBOT_ROOT/lerobot/.venv/bin/python3 $AGENTROBOT_ROOT/lerobot/policy_server.py \
  --checkpoint lerobot/pi05_libero_finetuned_v044 --port 18825 &
PID=$!

# Wait for server (may take several minutes due to model loading + warmup)
for i in $(seq 1 240); do
  curl -s http://localhost:18825/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

if ! curl -s http://localhost:18825/healthz >/dev/null 2>&1; then
  echo "ERROR: pi0.5 server never became ready"
  kill $PID 2>/dev/null
  exit 1
fi

echo ""
echo "=========================================="
echo "  EVAL: pi0.5 on libero_spatial ($(date))"
echo "=========================================="

PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18825 --policy pi05 --task_suite_name libero_spatial \
  --num_trials_per_task 5 --arm_controller cartesian_pose \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video

echo "pi05 spatial exit: $?"

kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "Done $(date)"
