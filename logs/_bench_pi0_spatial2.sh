#!/bin/bash
#SBATCH --job-name=bench-pi0
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --exclude=cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/bench-pi0-spatial2-%j.log

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface

echo "=== pi0 on LIBERO-spatial ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

cd $AGENTROBOT_ROOT/openpi
source .venv/bin/activate

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openpi/src \
  python3 $AGENTROBOT_ROOT/openpi/scripts/policy_server.py \
  --config pi0_libero --port 18820 &
SERVER_PID=$!

for i in $(seq 1 90); do
  curl -s http://localhost:18820/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }; sleep 10
done

if ! curl -s http://localhost:18820/healthz > /dev/null 2>&1; then
  echo "Server not ready after 15 min!"; kill $SERVER_PID 2>/dev/null; exit 1
fi

export MUJOCO_GL=egl
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18820 --policy pi0 --task_suite_name libero_spatial \
  --num_trials_per_task 5 --arm_controller joint_vel \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
echo "Exit: $?"
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
