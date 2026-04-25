#!/bin/bash
#SBATCH --job-name=v11-ovla-lib
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=06:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/v11-ovla-lib-%j.log

# ============================================================
#  v11: OpenVLA-OFT × LIBERO (per-suite finetuned checkpoints)
#  With preflight check + ActionSanityChecker
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
PORT=18880

echo "============================================"
echo "  v11: OpenVLA-OFT × LIBERO (per-suite checkpoints)"
echo "  $(date) | Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Each LIBERO suite requires its own finetuned checkpoint AND unnorm_key
declare -A CHECKPOINTS=(
  [libero_spatial]="moojink/openvla-7b-oft-finetuned-libero-spatial"
  [libero_object]="moojink/openvla-7b-oft-finetuned-libero-object"
  [libero_goal]="moojink/openvla-7b-oft-finetuned-libero-goal"
  [libero_10]="moojink/openvla-7b-oft-finetuned-libero-10"
)
declare -A UNNORM_KEYS=(
  [libero_spatial]="libero_spatial_no_noops"
  [libero_object]="libero_object_no_noops"
  [libero_goal]="libero_goal_no_noops"
  [libero_10]="libero_10_no_noops"
)

for SUITE in libero_object libero_goal libero_10; do
  CKPT=${CHECKPOINTS[$SUITE]}
  UNNORM=${UNNORM_KEYS[$SUITE]}
  echo ""
  echo "=========================================="
  echo "  EVAL: OpenVLA on LIBERO/$SUITE ($(date))"
  echo "  Checkpoint: $CKPT"
  echo "=========================================="

  # Start OpenVLA server with suite-specific checkpoint
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
    $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
    --pretrained_checkpoint "$CKPT" --unnorm_key "$UNNORM" --port $PORT &
  PID=$!

  for i in $(seq 1 120); do
    curl -s http://localhost:$PORT/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
    kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
  done

  if ! curl -s http://localhost:$PORT/healthz >/dev/null 2>&1; then
    echo "ERROR: Server never became ready"; kill $PID 2>/dev/null; continue
  fi

  # Run evaluation
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
    $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
    --policy_server_addr localhost:$PORT --policy openvla --task_suite_name "$SUITE" \
    --num_trials_per_task 5 --no_save_video \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results_v11 2>&1
  echo "  → OpenVLA on LIBERO/$SUITE: exit $?"

  # Stop server before loading next checkpoint
  kill $PID 2>/dev/null; wait $PID 2>/dev/null
  sleep 5  # let port fully release
done

echo ""
echo "============================================"
echo "  ALL EVALUATIONS COMPLETE — $(date)"
echo "============================================"
