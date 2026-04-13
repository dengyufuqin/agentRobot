#!/bin/bash
#SBATCH --job-name=eval-openvla
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.log

AGENT_ROOT=${AGENTROBOT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PORT=18800
CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial"
TASK_SUITE="libero_spatial"
NUM_TRIALS=5

OPENVLA_PYTHON=$AGENT_ROOT/openvla/.venv/bin/python3
LIBERO_PYTHON=$AGENT_ROOT/LIBERO/.venv/bin/python3

echo "=== OpenVLA LIBERO Eval ==="
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "Task suite: $TASK_SUITE, Trials: $NUM_TRIALS"

# Step 1: Start server
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENT_ROOT/agentic/policy_websocket/src:$AGENT_ROOT/openvla \
  $OPENVLA_PYTHON $AGENT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint $CHECKPOINT --port $PORT --execute_steps 1 &
SERVER_PID=$!

# Step 2: Wait for server
for i in $(seq 1 60); do
  curl -s http://localhost:$PORT/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
  sleep 10
done

# Step 3: Run eval (no set -e, capture exit code)
echo "=== Running eval ==="
export MUJOCO_GL=egl
PYTHONPATH=$AGENT_ROOT/agentic/policy_websocket/src:$AGENT_ROOT/LIBERO \
  $LIBERO_PYTHON -u $AGENT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:$PORT \
  --policy openvla-oft \
  --task_suite_name $TASK_SUITE \
  --num_trials_per_task $NUM_TRIALS \
  --arm_controller cartesian_pose \
  --log_dir $AGENT_ROOT/logs/eval_results \
  --no_save_video
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
