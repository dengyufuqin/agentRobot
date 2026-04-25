#!/bin/bash
#SBATCH --job-name=eval-svla-rc
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-svla-rc-%j.log

# ============================================================
#  SpatialVLA × RoboCasa (kitchen tasks)
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
PORT=18852

echo "============================================"
echo "  SpatialVLA × RoboCasa (v9)"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Start SpatialVLA server
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/SpatialVLA \
  $AGENTROBOT_ROOT/SpatialVLA/.venv/bin/python3 $AGENTROBOT_ROOT/SpatialVLA/policy_server.py \
  --checkpoint IPEC-COMMUNITY/spatialvla-4b-224-pt --unnorm_key none --port $PORT &
PID=$!

for i in $(seq 1 120); do
  curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

if ! curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
  echo "ERROR: SpatialVLA server never became ready"
  kill $PID 2>/dev/null
  exit 1
fi

echo ""
echo "=== RoboCasa Evaluations ==="

for TASK in PnPCounterToCab PnPCabToCounter PnPCounterToSink OpenSingleDoor CloseSingleDoor TurnOnSinkFaucet; do
  echo ""
  echo "=========================================="
  echo "  EVAL: SpatialVLA on RoboCasa/$TASK ($(date))"
  echo "=========================================="
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/robocasa \
    $AGENTROBOT_ROOT/robocasa/.venv/bin/python3 -u $AGENTROBOT_ROOT/robocasa/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy spatialvla --task_name "$TASK" \
    --num_trials 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results 2>&1
  echo "  → SpatialVLA on RoboCasa/$TASK: exit $?"
done

kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo ""
echo "============================================"
echo "  ALL EVALUATIONS COMPLETE"
echo "  $(date)"
echo "============================================"
