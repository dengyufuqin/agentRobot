#!/bin/bash
#SBATCH --job-name=bench-pi0.5
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/benchmark-pi0.5-maniskill:PickCube-v1-%j.log
#SBATCH --exclude=cn19,cn23

export LD_LIBRARY_PATH=$NVIDIA_BASE/cuda_runtime/lib:$NVIDIA_BASE/cublas/lib:$NVIDIA_BASE/cudnn/lib:$NVIDIA_BASE/cufft/lib:$NVIDIA_BASE/cusolver/lib:$NVIDIA_BASE/cusparse/lib:$NVIDIA_BASE/nvjitlink/lib:$NVIDIA_BASE/cuda_cupti/lib:/usr/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib64:$LIBRARY_PATH  # gcc link-time — Triton gcc wrapper needs -lcuda → /usr/lib64/libcuda.so
export TORCHDYNAMO_DISABLE=1  # skip torch.compile max-autotune (45+ min first-inference JIT) for fast verification
export TORCH_COMPILE_DISABLE=1

echo "=== Benchmark: pi0.5 on maniskill:PickCube-v1 ==="
echo "Eval client: maniskill, Task: PickCube-v1"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Start server — openpi server (JAX/orbax) handles pi05 safetensors format.
# Config pi05_libero matches action_dim=7 Franka for ManiSkill.
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
  PickCube-v1 \
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
