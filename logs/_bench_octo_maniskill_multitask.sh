#!/bin/bash
#SBATCH --job-name=octo-ms-multi
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/benchmark-octo-maniskill-multi-%j.log
#SBATCH --exclude=cn02,cn19,cn23

export LD_LIBRARY_PATH=$NVIDIA_BASE/cuda_runtime/lib:$NVIDIA_BASE/cublas/lib:$NVIDIA_BASE/cudnn/lib:$NVIDIA_BASE/cufft/lib:$NVIDIA_BASE/cusolver/lib:$NVIDIA_BASE/cusparse/lib:$NVIDIA_BASE/nvjitlink/lib:$NVIDIA_BASE/cuda_cupti/lib

echo "=== Octo × ManiSkill multi-task (RPD ckpt + human camera) ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/octo /mnt/vast/home/yd66byne/code/agentRobot/octo/.venv/bin/python3 /mnt/vast/home/yd66byne/code/agentRobot/octo/policy_server.py --port 18800 --checkpoint /mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/checkpoints/octo-maniskill &
SERVER_PID=$!

for i in $(seq 1 60); do
  curl -s http://localhost:18800/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
  sleep 10
done

for TASK in StackCube-v1 PushCube-v1 PullCubeTool-v1 PokeCube-v1 PegInsertionSide-v1 LiftPegUpright-v1 PlugCharger-v1; do
  echo ""
  echo "=========================================="
  echo "=== $TASK ==="
  echo "=========================================="
  PYTHONPATH=/mnt/vast/home/yd66byne/code/agentRobot/agentic/policy_websocket/src:/mnt/vast/home/yd66byne/code/agentRobot/ManiSkill /mnt/vast/home/yd66byne/code/agentRobot/ManiSkill/.venv/bin/python3 -u /mnt/vast/home/yd66byne/code/agentRobot/ManiSkill/scripts/run_eval.py \
    --policy_server_addr localhost:18800 \
    --policy octo \
    --env_id "$TASK" \
    --num_trials 10 \
    --log_dir /mnt/vast/home/yd66byne/code/agentRobot/logs/eval_results \
    --use_human_camera \
    --img_res 256 \
    --no_save_video
  echo "→ $TASK exit $?"
done

kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "=== DONE ==="
