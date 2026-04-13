#!/bin/bash
#SBATCH --job-name=bench-octo
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --exclude=cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/bench-octo-spatial-%j.log

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface

echo "=== Octo on LIBERO-spatial ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Octo needs special JAX/CUDA library setup
cd $AGENTROBOT_ROOT/octo
source .venv/bin/activate

NVIDIA_BASE=$(.venv/bin/python3 -c 'import nvidia; import os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null)
if [ -n "$NVIDIA_BASE" ]; then
  export LD_LIBRARY_PATH=${NVIDIA_BASE}/cuda_runtime/lib:${NVIDIA_BASE}/cublas/lib:${NVIDIA_BASE}/cudnn/lib:${NVIDIA_BASE}/cufft/lib:${NVIDIA_BASE}/cusolver/lib:${NVIDIA_BASE}/cusparse/lib:${NVIDIA_BASE}/nvjitlink/lib:${NVIDIA_BASE}/cuda_cupti/lib:${LD_LIBRARY_PATH}
fi
for cuda_path in /usr/local/cuda/bin /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin; do
  [ -d "$cuda_path" ] && export PATH=$cuda_path:$PATH && break
done

# Start Octo server
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/octo \
  python3 $AGENTROBOT_ROOT/octo/policy_server.py \
  --checkpoint "hf://rail-berkeley/octo-small-1.5" --port 18815 &
SERVER_PID=$!

for i in $(seq 1 60); do
  curl -s http://localhost:18815/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }; sleep 10
done

# Run eval (only 3 trials per task to save time since Octo is general model)
export MUJOCO_GL=egl
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18815 --policy octo --task_suite_name libero_spatial \
  --num_trials_per_task 3 --arm_controller cartesian_pose \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null
