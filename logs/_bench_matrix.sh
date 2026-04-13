#!/bin/bash
#SBATCH --job-name=bench-matrix
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/bench-matrix-%j.log

# ============================================================
# BENCHMARK MATRIX: Prove "any model × any benchmark"
# ============================================================
# Tests multiple model × benchmark combinations sequentially
# on a single GPU. Each test: start server → run eval → stop server.
#
# Matrix:
#   1. OpenVLA × LIBERO_spatial     (proven: 94%)
#   2. OpenVLA × ManiSkill:PickCube-v1
#   3. OpenVLA × RoboTwin:open_laptop
# ============================================================

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
LOG_DIR=$AGENTROBOT_ROOT/logs/eval_results

echo "============================================"
echo "  BENCHMARK MATRIX: any model × any benchmark"
echo "============================================"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

RESULTS=""

run_single_bench() {
    local LABEL="$1"
    local SERVER_PYTHON="$2"
    local SERVER_SCRIPT="$3"
    local SERVER_PP="$4"
    local SERVER_ARGS="$5"
    local SERVER_ENV="$6"
    local EVAL_PYTHON="$7"
    local EVAL_SCRIPT="$8"
    local EVAL_PP="$9"
    local EVAL_ARGS="${10}"
    local EVAL_ENV="${11}"

    echo ""
    echo "========================================"
    echo "  $LABEL"
    echo "========================================"

    # Start server
    echo "Starting server..."
    eval "$SERVER_ENV CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$SERVER_PP $SERVER_PYTHON $SERVER_SCRIPT $SERVER_ARGS" &
    local SERVER_PID=$!

    # Wait for server
    local PORT=$(echo "$SERVER_ARGS" | grep -oP '(?<=--port )\d+')
    [ -z "$PORT" ] && PORT=18800
    for i in $(seq 1 60); do
        curl -s http://localhost:$PORT/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
        kill -0 $SERVER_PID 2>/dev/null || { echo "Server died!"; RESULTS="$RESULTS\n$LABEL: SERVER_FAILED"; return 1; }
        sleep 10
    done

    # Run eval
    echo "Running eval..."
    eval "$EVAL_ENV PYTHONPATH=$EVAL_PP $EVAL_PYTHON -u $EVAL_SCRIPT $EVAL_ARGS"
    local EXIT=$?
    echo "Eval exit: $EXIT"

    # Extract success rate from log
    local SR=$(grep -oP '(?:overall|average|Success rate).*?(\d+\.?\d*)%' "$LOG_DIR"/*/eval.log 2>/dev/null | tail -1 | grep -oP '\d+\.?\d*%' | tail -1)
    [ -z "$SR" ] && SR="N/A"
    RESULTS="$RESULTS\n$LABEL: exit=$EXIT, success_rate=$SR"

    # Cleanup
    kill $SERVER_PID 2>/dev/null
    wait $SERVER_PID 2>/dev/null
    echo "Server stopped."
}

# ============================================================
# Test 1: OpenVLA × LIBERO_spatial (baseline, should be ~94%)
# ============================================================
run_single_bench \
    "OpenVLA × LIBERO_spatial" \
    "$AGENTROBOT_ROOT/openvla/.venv/bin/python3" \
    "$AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py" \
    "$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla" \
    "--pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --port 18800 --execute_steps 1" \
    "HF_HOME=$HOME/.cache/huggingface" \
    "$AGENTROBOT_ROOT/LIBERO/.venv/bin/python3" \
    "$AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py" \
    "$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO" \
    "--policy_server_addr localhost:18800 --policy openvla-oft --task_suite_name libero_spatial --num_trials_per_task 3 --arm_controller cartesian_pose --log_dir $LOG_DIR --no_save_video" \
    "MUJOCO_GL=egl"

# ============================================================
# Test 2: OpenVLA × ManiSkill:PickCube-v1
# ============================================================
run_single_bench \
    "OpenVLA × ManiSkill:PickCube-v1" \
    "$AGENTROBOT_ROOT/openvla/.venv/bin/python3" \
    "$AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py" \
    "$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla" \
    "--pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --port 18800 --execute_steps 1" \
    "HF_HOME=$HOME/.cache/huggingface" \
    "$AGENTROBOT_ROOT/ManiSkill/.venv/bin/python3" \
    "$AGENTROBOT_ROOT/ManiSkill/scripts/run_eval.py" \
    "$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/ManiSkill" \
    "--policy_server_addr localhost:18800 --policy openvla --env_id PickCube-v1 --num_trials 3 --log_dir $LOG_DIR --no_save_video" \
    ""

# ============================================================
# Test 3: OpenVLA × RoboTwin:open_laptop
# ============================================================
run_single_bench \
    "OpenVLA × RoboTwin:open_laptop" \
    "$AGENTROBOT_ROOT/openvla/.venv/bin/python3" \
    "$AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py" \
    "$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla" \
    "--pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial --port 18800 --execute_steps 1" \
    "HF_HOME=$HOME/.cache/huggingface" \
    "$AGENTROBOT_ROOT/RoboTwin/.venv/bin/python3" \
    "$AGENTROBOT_ROOT/RoboTwin/script/run_eval_ws.py" \
    "$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/RoboTwin" \
    "--policy_server_addr localhost:18800 --policy openvla --task_name open_laptop --action_type ee --num_trials 3 --log_dir $LOG_DIR --no_save_video" \
    "LD_LIBRARY_PATH=/mnt/vast/home/yd66byne/miniconda3/lib:\$LD_LIBRARY_PATH"

# ============================================================
# SUMMARY
# ============================================================
echo ""
echo "============================================"
echo "  BENCHMARK MATRIX RESULTS"
echo "============================================"
echo -e "$RESULTS"
echo "============================================"
echo "Done."
