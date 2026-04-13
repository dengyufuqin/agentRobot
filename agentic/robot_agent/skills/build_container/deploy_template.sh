#!/bin/bash
# Template for deploying a containerized policy server
# Usage: deploy_container.sh NODE PORT GPU_ID SIF_PATH [EXTRA_ARGS]

NODE=$1
PORT=${2:-18800}
GPU_ID=${3:-7}
SIF_PATH=$4
shift 4

LOG_DIR=${AGENTROBOT_ROOT:-.}/logs
REPO_NAME=$(basename "$SIF_PATH" .sif)
LOG="$LOG_DIR/${REPO_NAME}-${NODE}-${PORT}.log"

# Check port
if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 "$NODE" "ss -tlnp | grep :${PORT}" 2>/dev/null | grep -q ":${PORT}"; then
  echo "ERROR: Port $PORT already in use on $NODE"
  exit 1
fi

ssh -o StrictHostKeyChecking=no "$NODE" "
  export CUDA_VISIBLE_DEVICES=$GPU_ID
  nohup apptainer run --nv \
    --bind /mnt/vast:/mnt/vast \
    $SIF_PATH \
    --port $PORT $@ \
    > $LOG 2>&1 &
  echo \"PID=\$!\"
"

echo "Container starting on $NODE:$PORT (GPU $GPU_ID)"
echo "Log: $LOG"
