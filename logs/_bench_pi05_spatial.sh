#!/bin/bash
#SBATCH --job-name=eval-pi05
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-pi05-%j.log

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
echo "Node: $(hostname) | $(date)"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/lerobot/src \
  $AGENTROBOT_ROOT/lerobot/.venv/bin/python3 $AGENTROBOT_ROOT/lerobot/policy_server.py \
  --checkpoint lerobot/pi05_libero_finetuned_v044 --port 18814 &
PID=$!

for i in $(seq 1 120); do
  curl -s http://localhost:18814/healthz >/dev/null 2>&1 && { echo "Server ready (${i}0s)"; break; }
  kill -0 $PID 2>/dev/null || { echo "Server died!"; exit 1; }; sleep 10
done

PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18814 --policy pi05 --task_suite_name libero_spatial \
  --num_trials_per_task 5 --arm_controller cartesian_pose \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
echo "pi05 spatial exit: $?"

kill $PID 2>/dev/null; wait $PID 2>/dev/null; echo "Done $(date)"
