#!/bin/bash
#SBATCH --job-name=eval-debug
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.log

AGENT_ROOT=${AGENTROBOT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}
PORT=18800
CHECKPOINT="moojink/openvla-7b-oft-finetuned-libero-spatial"

OPENVLA_PYTHON=$AGENT_ROOT/openvla/.venv/bin/python3
LIBERO_PYTHON=$AGENT_ROOT/LIBERO/.venv/bin/python3

echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Start server
echo "Starting server..."
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENT_ROOT/agentic/policy_websocket/src:$AGENT_ROOT/openvla \
  $OPENVLA_PYTHON $AGENT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint $CHECKPOINT --port $PORT --execute_steps 1 &
SERVER_PID=$!

# Wait for server
for i in $(seq 1 60); do
  curl -s http://localhost:$PORT/healthz > /dev/null 2>&1 && break
  kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
  sleep 10
done
echo "Server ready"

# Run debug eval
echo "Running debug eval..."
export MUJOCO_GL=egl
PYTHONPATH=$AGENT_ROOT/agentic/policy_websocket/src:$AGENT_ROOT/LIBERO \
  $LIBERO_PYTHON -u $AGENT_ROOT/agentic/scripts/test_eval_debug.py localhost $PORT

echo "Debug eval exit: $?"
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
