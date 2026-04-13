#!/bin/bash
#SBATCH --job-name=bench-v3
#SBATCH --partition=all
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=32
#SBATCH --mem=300G
#SBATCH --time=08:00:00
#SBATCH --nodelist=cn21
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/bench-v3-%j.log

# ============================================================
#  Benchmark v3: pi0.5 × LIBERO spatial + Octo × 4 LIBERO suites
#  3 GPUs on cn21
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl

echo "============================================"
echo "  BENCHMARK v3 — pi0.5 + Octo × LIBERO"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

PIDS=()

# ── GPU 0: LeRobot pi0.5 server ──────────────────────────
echo "=== [GPU 0] LeRobot pi0.5 × libero_spatial ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/lerobot/src \
  $AGENTROBOT_ROOT/lerobot/.venv/bin/python3 $AGENTROBOT_ROOT/lerobot/policy_server.py \
  --checkpoint lerobot/pi05_libero_finetuned_v044 --port 18820 &
PIDS+=($!)

# ── GPU 1: Octo server (for all LIBERO suites) ───────────
echo "=== [GPU 1] Octo × LIBERO (all suites) ==="
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/octo \
  $AGENTROBOT_ROOT/octo/.venv/bin/python3 $AGENTROBOT_ROOT/octo/policy_server.py \
  --port 18821 &
PIDS+=($!)

# ── Wait for servers ─────────────────────────────────────
echo ""
echo "=== Waiting for servers ==="
PORTS=(18820 18821)
for idx in "${!PORTS[@]}"; do
  port=${PORTS[$idx]}
  pid=${PIDS[$idx]}
  for i in $(seq 1 180); do
    curl -s http://localhost:$port/healthz > /dev/null 2>&1 && { echo "  Port $port ready after ${i}0s"; break; }
    kill -0 $pid 2>/dev/null || { echo "  Port $port server DIED (pid $pid)!"; break; }
    sleep 10
  done
done
echo ""

# Verify servers
for port in "${PORTS[@]}"; do
  if ! curl -s http://localhost:$port/healthz > /dev/null 2>&1; then
    echo "WARNING: Server on port $port is NOT responding!"
  fi
done

echo ""
echo "=== Starting evaluations ==="

run_eval() {
  local name=$1 port=$2 policy=$3 suite=$4 controller=$5
  echo ""
  echo "=========================================="
  echo "  EVAL: $name on $suite ($(date))"
  echo "=========================================="

  if ! curl -s http://localhost:$port/healthz > /dev/null 2>&1; then
    echo "SKIP: server on port $port not available"
    return 1
  fi

  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
    $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
    --policy_server_addr localhost:$port --policy "$policy" --task_suite_name "$suite" \
    --num_trials_per_task 5 --arm_controller "$controller" \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video

  local exit_code=$?
  echo "  → $name on $suite: exit $exit_code"
  return $exit_code
}

# ── pi0.5 × libero_spatial ─────────────────────────────
run_eval "pi05" 18820 "pi05" "libero_spatial" "cartesian_pose"

# ── Octo × 4 LIBERO suites (sequential on same server) ─
run_eval "Octo" 18821 "octo" "libero_spatial" "cartesian_pose"
run_eval "Octo" 18821 "octo" "libero_object"  "cartesian_pose"
run_eval "Octo" 18821 "octo" "libero_goal"    "cartesian_pose"
run_eval "Octo" 18821 "octo" "libero_10"      "cartesian_pose"

# ── Cleanup ────────────────────────────────────────────
echo ""
echo "=== Cleanup ==="
for pid in "${PIDS[@]}"; do
  kill $pid 2>/dev/null
done
wait "${PIDS[@]}" 2>/dev/null

echo ""
echo "============================================"
echo "  BENCHMARK v3 COMPLETE"
echo "  $(date)"
echo "============================================"
