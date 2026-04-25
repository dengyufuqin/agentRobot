#!/bin/bash
#SBATCH --job-name=bench-rdt
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/benchmark-rdt-maniskill:PlugCharger-v1-%j.log





echo "=== Benchmark: rdt on maniskill:PlugCharger-v1 ==="
echo "Eval client: maniskill, Task: PlugCharger-v1"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Start server (CUDA_VISIBLE_DEVICES as prefix, not global)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/lerobot /mnt/vast/home/yd66byne/code/agentRobot/lerobot/.venv/bin/python3 /mnt/vast/home/yd66byne/code/agentRobot/lerobot/policy_server.py --checkpoint /mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/checkpoints/rdt-maniskill --port 18800 &
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
  lerobot \
  --env_id \
  PlugCharger-v1 \
  --num_trials \
  10 \
  --log_dir \
  /mnt/vast/home/yd66byne/code/agentRobot/logs/eval_results \
  --no_save_video \
  --max_episode_steps \
  300 \
  --img_res \
  384 \
  --use_human_camera \
  --control_mode \
  pd_ee_delta_pose
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $EVAL_EXIT
