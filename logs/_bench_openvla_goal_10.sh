#!/bin/bash
#SBATCH --job-name=eval-vla-gt
#SBATCH --partition=all
#SBATCH --gres=gpu:2
#SBATCH --time=06:00:00
#SBATCH --exclude=cn19,cn23
#SBATCH --output=/mnt/vast/home/yd66byne/code/agentRobot/logs/eval-vla-gt-%j.log

export AGENTROBOT_ROOT=/mnt/vast/home/yd66byne/code/agentRobot
export HF_HOME=$HOME/.cache/huggingface
export MUJOCO_GL=egl
echo "Node: $(hostname) | $(date)"

# GPU 0: OpenVLA × libero_goal
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
  --unnorm_key libero_goal_no_noops --port 18812 --execute_steps 1 &
PID1=$!

# GPU 1: OpenVLA × libero_10
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/openvla \
  $AGENTROBOT_ROOT/openvla/.venv/bin/python3 $AGENTROBOT_ROOT/openvla/vla-scripts/policy_server.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
  --unnorm_key libero_10_no_noops --port 18813 --execute_steps 1 &
PID2=$!

for port in 18812 18813; do
  for i in $(seq 1 120); do
    curl -s http://localhost:$port/healthz >/dev/null 2>&1 && { echo "Port $port ready (${i}0s)"; break; }; sleep 10
  done
done

PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18812 --policy openvla-oft --task_suite_name libero_goal \
  --num_trials_per_task 5 --arm_controller cartesian_pose \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
echo "goal exit: $?"

PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/LIBERO \
  $AGENTROBOT_ROOT/LIBERO/.venv/bin/python3 -u $AGENTROBOT_ROOT/LIBERO/scripts/run_eval.py \
  --policy_server_addr localhost:18813 --policy openvla-oft --task_suite_name libero_10 \
  --num_trials_per_task 5 --arm_controller cartesian_pose \
  --log_dir $AGENTROBOT_ROOT/logs/eval_results --no_save_video
echo "libero_10 exit: $?"

kill $PID1 $PID2 2>/dev/null; wait $PID1 $PID2 2>/dev/null; echo "Done $(date)"
