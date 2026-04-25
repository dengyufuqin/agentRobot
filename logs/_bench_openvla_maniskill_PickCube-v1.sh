#!/bin/bash
#SBATCH --job-name=bench-openvla
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/benchmark-openvla-maniskill:PickCube-v1-%j.log

#SBATCH --exclude=cn16,cn17,cn19

export HF_HOME=/mnt/vast/home/yd66byne/.cache/huggingface

echo "=== Benchmark: openvla on maniskill:PickCube-v1 ==="
echo "Eval client: maniskill, Task: PickCube-v1"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Start server (CUDA_VISIBLE_DEVICES as prefix, not global)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/openvla /mnt/vast/home/yd66byne/code/agentRobot/openvla/.venv/bin/python3 /mnt/vast/home/yd66byne/code/agentRobot/openvla/vla-scripts/policy_server.py --pretrained_checkpoint /mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/checkpoints/openvla-maniskill --port 19992 --execute_steps 1 --unnorm_key maniskill_human:7.0.0 --num_images_in_input 1 --no_proprio  --no_flip_image --no_center_crop --no_invert_gripper &
SERVER_PID=$!

# Step 2: Wait for server
for i in $(seq 1 60); do
  curl -s http://localhost:19992/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
  sleep 10
done

# Step 3: Run eval
echo "=== Running eval (maniskill) ==="

PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/ManiSkill /mnt/vast/home/yd66byne/code/agentRobot/ManiSkill/.venv/bin/python3 -u /mnt/vast/home/yd66byne/code/agentRobot/ManiSkill/scripts/run_eval.py \
  --policy_server_addr \
  localhost:19992 \
  --policy \
  openvla-oft \
  --env_id \
  PickCube-v1 \
  --num_trials \
  5 \
  --log_dir \
  /mnt/vast/home/yd66byne/code/agentRobot/logs/eval_results \
  --no_save_video \
  --max_episode_steps \
  300 \
  --img_res \
  256 \
  --use_human_camera \
  --control_mode \
  pd_ee_delta_pose
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $EVAL_EXIT
