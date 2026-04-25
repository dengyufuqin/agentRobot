#!/bin/bash
#SBATCH --job-name=bench-smolvla
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/benchmark-smolvla-libero_object-%j.log
#SBATCH --nodelist=cn26




echo "=== Benchmark: smolvla on libero_object ==="
echo "Eval client: libero, Task: libero_object"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Start server (CUDA_VISIBLE_DEVICES as prefix, not global)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/lerobot /mnt/vast/home/yd66byne/code/agentRobot/lerobot/.venv/bin/python3 /mnt/vast/home/yd66byne/code/agentRobot/lerobot/policy_server.py --checkpoint /mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/checkpoints/smolvla-libero --port 18800 &
SERVER_PID=$!

# Step 2: Wait for server
for i in $(seq 1 60); do
  curl -s http://localhost:18800/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
  sleep 10
done

# Step 3: Run eval
echo "=== Running eval (libero) ==="
export MUJOCO_GL=egl
PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/LIBERO /mnt/vast/home/yd66byne/code/agentRobot/LIBERO/.venv/bin/python3 -u /mnt/vast/home/yd66byne/code/agentRobot/LIBERO/scripts/run_eval.py \
  --policy_server_addr \
  localhost:18800 \
  --policy \
  lerobot \
  --task_suite_name \
  libero_object \
  --num_trials_per_task \
  10 \
  --arm_controller \
  cartesian_pose \
  --log_dir \
  /mnt/vast/home/yd66byne/code/agentRobot/logs/eval_results \
  --no_save_video
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $EVAL_EXIT
