#!/bin/bash
#SBATCH --job-name=bench-openvla
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/benchmark-openvla-libero_spatial-%j.log


export HF_HOME=/mnt/vast/home/yd66byne/.cache/huggingface

echo "=== Benchmark: openvla on libero_spatial ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Start server (CUDA_VISIBLE_DEVICES as prefix, not global)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/openvla /mnt/vast/home/yd66byne/code/agentRobot/openvla/.venv/bin/python3 /mnt/vast/home/yd66byne/code/agentRobot/openvla/vla-scripts/policy_server.py --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --port 18800 --execute_steps 1 &
SERVER_PID=$!

# Step 2: Wait for server
for i in $(seq 1 60); do
  curl -s http://localhost:18800/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
  sleep 10
done

# Step 3: Run eval (MUJOCO_GL set here, not globally)
echo "=== Running eval ==="
export MUJOCO_GL=egl
PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/LIBERO /mnt/vast/home/yd66byne/code/agentRobot/LIBERO/.venv/bin/python3 -u /mnt/vast/home/yd66byne/code/agentRobot/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18800 \
  --policy openvla-oft \
  --task_suite_name libero_spatial \
  --num_trials_per_task 5 \
  --arm_controller cartesian_pose \
  --log_dir /mnt/vast/home/yd66byne/code/agentRobot/logs/eval_results \
  --no_save_video
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $EVAL_EXIT
