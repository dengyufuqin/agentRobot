#!/bin/bash
#SBATCH --job-name=v11-demo-block
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:15:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/v11-demo-block-%j.log

# ============================================================
#  DEMO: Preflight system blocks bad eval (Octo × LIBERO, no finetune)
#  This job should EXIT EARLY with a clear "NEEDS FINETUNE" message
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
PORT=18885

echo "============================================"
echo "  DEMO: Preflight blocks Octo × LIBERO (no finetune)"
echo "  $(date) | Node: $(hostname)"
echo "============================================"

# Start Octo server
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/octo \
  $AGENTROBOT_ROOT/octo/.venv/bin/python3 $AGENTROBOT_ROOT/octo/policy_server.py \
  --port $PORT &
PID=$!

for i in $(seq 1 60); do
  curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

if ! curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
  echo "ERROR: Server never became ready"; kill $PID 2>/dev/null; exit 1
fi

echo ""
echo "=== TEST 1: Without --allow_cross_domain (should be BLOCKED) ==="
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:$PORT --policy octo --task_suite_name libero_spatial \
  --num_trials_per_task 1 --no_save_video \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results_v11 2>&1
echo "  → Exit code: $? (expected: 0, but 0 trials run)"

echo ""
echo "=== TEST 2: With --skip_preflight (should RUN) ==="
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:$PORT --policy octo --task_suite_name libero_spatial \
  --num_trials_per_task 1 --no_save_video --skip_preflight \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results_v11 2>&1
echo "  → Exit code: $?"

kill $PID 2>/dev/null; wait $PID 2>/dev/null

echo ""
echo "============================================"
echo "  DEMO COMPLETE — $(date)"
echo "============================================"
