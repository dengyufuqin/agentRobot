#!/bin/bash
#SBATCH --job-name=bench-octo
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/benchmark-octo-maniskill:PushCube-v1-%j.log

#SBATCH --exclude=cn16,cn17,cn19

export LD_LIBRARY_PATH=$NVIDIA_BASE/cuda_runtime/lib:$NVIDIA_BASE/cublas/lib:$NVIDIA_BASE/cudnn/lib:$NVIDIA_BASE/cufft/lib:$NVIDIA_BASE/cusolver/lib:$NVIDIA_BASE/cusparse/lib:$NVIDIA_BASE/nvjitlink/lib:$NVIDIA_BASE/cuda_cupti/lib

echo "=== Benchmark: octo on maniskill:PushCube-v1 ==="
echo "Eval client: maniskill, Task: PushCube-v1"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Start server (CUDA_VISIBLE_DEVICES as prefix, not global)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/octo /mnt/vast/home/yd66byne/code/agentRobot/octo/.venv/bin/python3 /mnt/vast/home/yd66byne/code/agentRobot/octo/policy_server.py --port 25465 --checkpoint /mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/checkpoints/octo-maniskill &
SERVER_PID=$!

# Step 2: Wait for server
for i in $(seq 1 60); do
  curl -s http://localhost:25465/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
  sleep 10
done

# Step 3: Run eval
echo "=== Running eval (maniskill) ==="

PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/ManiSkill /mnt/vast/home/yd66byne/code/agentRobot/ManiSkill/.venv/bin/python3 -u /mnt/vast/home/yd66byne/code/agentRobot/ManiSkill/scripts/run_eval.py \
  --policy_server_addr \
  localhost:25465 \
  --policy \
  octo \
  --env_id \
  PushCube-v1 \
  --num_trials \
  50 \
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
