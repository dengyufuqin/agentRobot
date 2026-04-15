---
name: train_and_eval
description: "Full pipeline: finetune a model on benchmark data, then evaluate the resulting checkpoint — all in one SLURM job. No manual intervention needed."
version: 1.0.0
category: pipeline
parameters:
  policy:
    type: string
    description: "Policy to finetune and evaluate: pi0, pi0.5, openvla"
    required: true
  benchmark:
    type: string
    description: "Target benchmark. Training data is auto-resolved. Eval runs on the same benchmark. Examples: libero_spatial, libero_object, libero_goal, libero_10"
    required: true
  train_steps:
    type: integer
    description: "Number of finetuning steps"
    default: 50000
  batch_size:
    type: integer
    description: "Training batch size per GPU"
    default: 8
  learning_rate:
    type: string
    description: "Learning rate"
    default: "2e-5"
  num_eval_trials:
    type: integer
    description: "Number of evaluation trials per task"
    default: 5
  node:
    type: string
    description: "SLURM node (e.g. cn13). Auto-selected if omitted."
    required: false
requires:
  bins: [python3]
timeout: 120
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/train_and_eval/run_train_and_eval.py \
    --policy "{policy}" \
    --benchmark "{benchmark}" \
    --train_steps {train_steps} \
    --batch_size {batch_size} \
    --learning_rate {learning_rate} \
    --num_eval_trials {num_eval_trials} \
    $([ -n "{node}" ] && echo "--node {node}")
---

# Train and Evaluate (Full Pipeline)

One-command pipeline: finetune a VLA model on a benchmark's training data, then evaluate the resulting checkpoint on the same benchmark. Runs as a single SLURM job.

## Pipeline Steps

1. **Finetune** — Train the model on benchmark data (LeRobot or OpenVLA backend)
2. **Deploy** — Start the policy server with the new checkpoint
3. **Evaluate** — Run the benchmark eval client against the server
4. **Report** — Print success rate and save results

## Examples

```
# Finetune pi0 on LIBERO spatial, then evaluate
train_and_eval(policy="pi0", benchmark="libero_spatial", train_steps=50000)

# Quick test: 1000 steps, 3 trials
train_and_eval(policy="pi0.5", benchmark="libero_object", train_steps=1000, num_eval_trials=3)
```

## Output

- Checkpoint: `$AGENTROBOT_ROOT/checkpoints/<policy>-<benchmark>-<timestamp>/`
- Eval results in the SLURM log
- Success rate printed at the end
