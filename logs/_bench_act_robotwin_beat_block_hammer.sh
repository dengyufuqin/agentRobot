#!/bin/bash
#SBATCH --job-name=bench-act
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/benchmark-act-robotwin:beat_block_hammer-%j.log

#SBATCH --exclude=cn16,cn17,cn19



echo "=== Benchmark: act on robotwin:beat_block_hammer ==="
echo "Eval client: robotwin, Task: beat_block_hammer"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Start server (CUDA_VISIBLE_DEVICES as prefix, not global)
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/RoboTwin/policy/ACT /mnt/vast/home/yd66byne/code/agentRobot/RoboTwin/policy/ACT/.venv/bin/python3 /mnt/vast/home/yd66byne/code/agentRobot/RoboTwin/policy/ACT/policy_server.py --checkpoint /mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/checkpoints/robotwin_ckpts/act_avada11_beat_block_hammer/beat_block_hammer.click_bell/demo_clean-100/100 --port 19446 --task beat_block_hammer --action_dim 14 --chunk_size 50 &
SERVER_PID=$!

# Step 2: Wait for server
for i in $(seq 1 60); do
  curl -s http://localhost:19446/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
  sleep 10
done

# Step 3: Run eval
echo "=== Running eval (robotwin) ==="
export CUDA_HOME=/mnt/vast/spack/v0.23/opt/spack/linux-rocky9-sapphirerapids/gcc-13.3.0/cuda-12.6.2-iipq3kx6jniy56k3iqzxqlccmnl4tgt7
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_EXTENSIONS_DIR=/mnt/vast/home/yd66byne/code/agentRobot/RoboTwin/.venv/curobo_ext_cache
export TORCH_CUDA_ARCH_LIST="8.0 9.0"
PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/RoboTwin /mnt/vast/home/yd66byne/code/agentRobot/RoboTwin/.venv/bin/python3 -u /mnt/vast/home/yd66byne/code/agentRobot/RoboTwin/script/run_eval_ws.py \
  --policy_server_addr \
  localhost:19446 \
  --policy \
  act \
  --task_name \
  beat_block_hammer \
  --action_type \
  qpos \
  --num_trials \
  50 \
  --log_dir \
  /mnt/vast/home/yd66byne/code/agentRobot/logs/eval_results \
  --no_save_video
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $EVAL_EXIT
