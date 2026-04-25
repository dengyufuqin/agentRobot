#!/bin/bash
#SBATCH --job-name=bench-openpi
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/benchmark-openpi-maniskill:StackCube-v1-%j.log





echo "=== Benchmark: openpi on maniskill:StackCube-v1 ==="
echo "Eval client: maniskill, Task: StackCube-v1"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Start server (CUDA_VISIBLE_DEVICES as prefix, not global)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/openpi/src /mnt/vast/home/yd66byne/code/agentRobot/openpi/.venv/bin/python3 /mnt/vast/home/yd66byne/code/agentRobot/openpi/scripts/policy_server.py --config pi05_libero --checkpoint /mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/checkpoints/pi05-maniskill --port 18800 &
SERVER_PID=$!

# Step 2: Wait for server
for i in $(seq 1 60); do
  curl -s http://localhost:18800/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
  sleep 10
done

# Step 3: Run eval
echo "=== Running eval (maniskill) ==="

PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/ManiSkill /mnt/vast/home/yd66byne/code/agentRobot/ManiSkill/.venv/bin/python3 -u /mnt/vast/home/yd66byne/code/agentRobot/ManiSkill/scripts/run_eval.py \
  --policy_server_addr \
  localhost:18800 \
  --policy \
  openpi \
  --env_id \
  StackCube-v1 \
  --num_trials \
  10 \
  --log_dir \
  /mnt/vast/home/yd66byne/code/agentRobot/logs/eval_results \
  --no_save_video
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $EVAL_EXIT
