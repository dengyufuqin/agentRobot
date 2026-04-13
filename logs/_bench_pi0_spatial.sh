#!/bin/bash
#SBATCH --job-name=bench-pi0
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/bench-pi0-spatial-%j.log

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface

echo "=== pi0 on LIBERO-spatial ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd $AGENTROBOT_ROOT/openpi
source .venv/bin/activate

# Start pi0 server with LIBERO config
echo "Starting pi0 server (config=pi0_libero)..."
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openpi/src \
  python3 $AGENTROBOT_ROOT/openpi/scripts/policy_server.py \
  --config pi0_libero \
  --port 18820 &
SERVER_PID=$!

# pi0 takes longer to download checkpoint + JIT compile, wait up to 15min
for i in $(seq 1 90); do
  curl -s http://localhost:18820/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server process died"; cat $AGENTROBOT_ROOT/logs/bench-pi0-spatial-${SLURM_JOB_ID}.log | tail -30; exit 1; }
  sleep 10
done

# Check if server is actually ready
if ! curl -s http://localhost:18820/healthz > /dev/null 2>&1; then
  echo "Server not ready after 15 minutes!"
  kill $SERVER_PID 2>/dev/null
  exit 1
fi

# Run LIBERO eval
# pi0 uses joint_velocity action space (--arm_controller joint_vel)
echo "=== Running eval ==="
export MUJOCO_GL=egl
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18820 --policy pi0 --task_suite_name libero_spatial \
  --num_trials_per_task 5 --arm_controller joint_vel \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
