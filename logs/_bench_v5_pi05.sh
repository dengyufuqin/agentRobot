#!/bin/bash
#SBATCH --job-name=eval-pi05v3
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-pi05v3-%j.log

# ============================================================
#  pi0.5 × LIBERO (all 4 suites) — v3: fixed state mapping
#  Fix: uses eef_pos + axis_angle(eef_quat) + gripper_qpos
#        instead of joint_pos + gripper[:1]
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
PORT=18825

echo "============================================"
echo "  pi0.5 × LIBERO (v3 — fixed state mapping)"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Start pi0.5 server
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/lerobot/src \
  $AGENTROBOT_ROOT/lerobot/.venv/bin/python3 $AGENTROBOT_ROOT/lerobot/policy_server.py \
  --checkpoint lerobot/pi05_libero_finetuned_v044 --port $PORT &
PID=$!

# Wait for server (may take several minutes due to model loading + JIT warmup)
for i in $(seq 1 240); do
  curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

if ! curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
  echo "ERROR: pi0.5 server never became ready"
  kill $PID 2>/dev/null
  exit 1
fi

for SUITE in libero_spatial libero_object libero_goal libero_10; do
  echo ""
  echo "=========================================="
  echo "  EVAL: pi0.5 on $SUITE ($(date))"
  echo "=========================================="

  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
    $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy pi05 --task_suite_name $SUITE \
    --num_trials_per_task 5 --arm_controller cartesian_pose \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video

  echo "  → pi0.5 on $SUITE: exit $?"
done

kill $PID 2>/dev/null; wait $PID 2>/dev/null
echo "Done $(date)"
