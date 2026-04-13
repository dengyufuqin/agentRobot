#!/bin/bash
#SBATCH --job-name=bench-v2
#SBATCH --partition=all
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=10:00:00
#SBATCH --nodelist=cn21
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/bench-v2-%j.log

# ============================================================
#  LIBERO Benchmark v2
#  OpenVLA × 4 suites + pi0.5 × spatial on 5 GPUs
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl

echo "============================================"
echo "  LIBERO BENCHMARK v2"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "============================================"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

PIDS=()

# GPU 0: OpenVLA × libero_spatial
echo "=== [GPU 0] OpenVLA × libero_spatial ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --unnorm_key libero_spatial_no_noops \
  --port 18810 --execute_steps 1 &
PIDS+=($!)

# GPU 1: OpenVLA × libero_object
echo "=== [GPU 1] OpenVLA × libero_object ==="
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
  --unnorm_key libero_object_no_noops \
  --port 18811 --execute_steps 1 &
PIDS+=($!)

# GPU 2: OpenVLA × libero_goal
echo "=== [GPU 2] OpenVLA × libero_goal ==="
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
  --unnorm_key libero_goal_no_noops \
  --port 18812 --execute_steps 1 &
PIDS+=($!)

# GPU 3: OpenVLA × libero_10
echo "=== [GPU 3] OpenVLA × libero_10 ==="
CUDA_VISIBLE_DEVICES=3 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --unnorm_key libero_10_no_noops \
  --port 18813 --execute_steps 1 &
PIDS+=($!)

# GPU 4: LeRobot pi0.5 × libero_spatial
echo "=== [GPU 4] LeRobot pi0.5 × libero_spatial ==="
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/lerobot/src \
  $AGENTROBOT_ROOT/lerobot/.venv/bin/python3 $AGENTROBOT_ROOT/lerobot/policy_server.py \
  --checkpoint lerobot/pi05_libero_finetuned_v044 --port 18814 &
PIDS+=($!)

# Wait for all servers
echo ""
echo "=== Waiting for all 5 servers ==="
PORTS=(18810 18811 18812 18813 18814)
for idx in "${!PORTS[@]}"; do
  port=${PORTS[$idx]}
  pid=${PIDS[$idx]}
  for i in $(seq 1 120); do
    curl -s http://localhost:$port/healthz > /dev/null 2>&1 && { echo "  Port $port ready after ${i}0s"; break; }
    kill -0 $pid 2>/dev/null || { echo "  Port $port server DIED (pid $pid)!"; break; }
    sleep 10
  done
done
echo ""

# Verify all servers
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

# OpenVLA × 4 suites
run_eval "OpenVLA" 18810 "openvla-oft" "libero_spatial" "cartesian_pose"
run_eval "OpenVLA" 18811 "openvla-oft" "libero_object"  "cartesian_pose"
run_eval "OpenVLA" 18812 "openvla-oft" "libero_goal"    "cartesian_pose"
run_eval "OpenVLA" 18813 "openvla-oft" "libero_10"      "cartesian_pose"

# pi0.5 × spatial
run_eval "pi05" 18814 "pi05" "libero_spatial" "cartesian_pose"

# Cleanup
echo ""
echo "=== Cleanup ==="
for pid in "${PIDS[@]}"; do
  kill $pid 2>/dev/null
done
wait "${PIDS[@]}" 2>/dev/null

echo ""
echo "============================================"
echo "  BENCHMARK COMPLETE"
echo "  $(date)"
echo "============================================"
