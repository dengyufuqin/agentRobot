#!/bin/bash
#SBATCH --job-name=mega-bench
#SBATCH --partition=all
#SBATCH --gres=gpu:3
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/mega-bench-%j.log

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface

echo "============================================"
echo "  MEGA BENCHMARK: Multiple Algorithms"
echo "============================================"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# =============================================
# Benchmark 1: OpenVLA on libero_object (GPU 0)
# =============================================
echo "=== [1/4] OpenVLA on libero_object (GPU 0) ==="
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object --port 18810 --execute_steps 1 &
PID1=$!

# =============================================
# Benchmark 2: OpenVLA on libero_goal (GPU 1)
# =============================================
echo "=== [2/4] OpenVLA on libero_goal (GPU 1) ==="
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal --port 18811 --execute_steps 1 &
PID2=$!

# =============================================
# Benchmark 3: OpenVLA on libero_10 (GPU 2)
# =============================================
echo "=== [3/4] OpenVLA on libero_10 (GPU 2) ==="
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 --port 18812 --execute_steps 1 &
PID3=$!

# Octo removed — JAX PTX compilation fails on H100 (sm_90)

# Wait for all servers to be ready
echo ""
echo "=== Waiting for servers ==="
for port_pid in "18810:PID1" "18811:PID2" "18812:PID3"; do
  port=${port_pid%%:*}
  for i in $(seq 1 90); do
    curl -s http://localhost:$port/healthz > /dev/null 2>&1 && { echo "Server on port $port ready after ${i}0s"; break; }
    sleep 10
  done
done

echo ""
echo "=== All servers launched, starting evals ==="
export MUJOCO_GL=egl

# Run evals SEQUENTIALLY (they share the same LIBERO venv and EGL context)
echo ""
echo "=== Eval 1: OpenVLA on libero_object ==="
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18810 --policy openvla-oft --task_suite_name libero_object \
  --num_trials_per_task 5 --arm_controller cartesian_pose \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
echo "Exit: $?"

echo ""
echo "=== Eval 2: OpenVLA on libero_goal ==="
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18811 --policy openvla-oft --task_suite_name libero_goal \
  --num_trials_per_task 5 --arm_controller cartesian_pose \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
echo "Exit: $?"

echo ""
echo "=== Eval 3: OpenVLA on libero_10 ==="
PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18812 --policy openvla-oft --task_suite_name libero_10 \
  --num_trials_per_task 5 --arm_controller cartesian_pose \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
echo "Exit: $?"

# Cleanup
echo ""
echo "=== Cleanup ==="
kill $PID1 $PID2 $PID3 2>/dev/null
wait $PID1 $PID2 $PID3 2>/dev/null
echo "Done."
