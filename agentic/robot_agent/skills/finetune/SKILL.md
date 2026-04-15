---
name: finetune
description: "Finetune a policy model on a benchmark dataset. Supports LeRobot models (pi0, pi0.5, smolvla) and OpenVLA. Submits as SLURM job for multi-GPU training."
version: 1.0.0
category: train
parameters:
  policy:
    type: string
    description: "Policy to finetune: pi0, pi0.5, smolvla, openvla"
    required: true
  benchmark:
    type: string
    description: "Target benchmark dataset. LIBERO: libero_spatial, libero_object, libero_goal, libero_10. RoboCasa: robocasa. Also accepts HuggingFace dataset repo_id directly."
    required: true
  base_checkpoint:
    type: string
    description: "Base model checkpoint to finetune from. Auto-resolved if omitted (e.g. lerobot/pi0_base for pi0)."
    required: false
  steps:
    type: integer
    description: "Number of training steps"
    default: 50000
  batch_size:
    type: integer
    description: "Batch size per GPU"
    default: 8
  num_gpus:
    type: integer
    description: "Number of GPUs for distributed training"
    default: 1
  learning_rate:
    type: string
    description: "Learning rate (e.g. 5e-4, 2e-5)"
    default: "2e-5"
  output_name:
    type: string
    description: "Name for the output checkpoint directory. Auto-generated if omitted."
    required: false
  node:
    type: string
    description: "SLURM node to run on (e.g. cn13). Auto-selected if omitted."
    required: false
  submit:
    type: string
    description: "If 'true', submit as SLURM job (recommended). If 'false', run locally."
    default: "true"
requires:
  bins: [python3]
timeout: 120
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/finetune/run_finetune.py \
    --policy "{policy}" \
    --benchmark "{benchmark}" \
    $([ -n "{base_checkpoint}" ] && echo "--base_checkpoint {base_checkpoint}") \
    --steps {steps} \
    --batch_size {batch_size} \
    --num_gpus {num_gpus} \
    --learning_rate {learning_rate} \
    $([ -n "{output_name}" ] && echo "--output_name {output_name}") \
    $([ -n "{node}" ] && echo "--node {node}") \
    $([ "{submit}" = "true" ] && echo "--submit")
---

# Finetune Policy Model

Finetune a VLA model on a benchmark's training data. Supports two backends:

## Supported Models

| Model | Backend | Base Checkpoint |
|-------|---------|----------------|
| pi0 | LeRobot | `lerobot/pi0_base` |
| pi0.5 | LeRobot | `lerobot/pi05_base` |
| smolvla | LeRobot | `lerobot/smolvla_base` |
| openvla | OpenVLA | `openvla/openvla-7b` |

## Supported Datasets

| Benchmark | LeRobot Dataset | OpenVLA (RLDS) Dataset |
|-----------|----------------|----------------------|
| libero_spatial | `lerobot/libero_spatial_image` | `libero_spatial_no_noops` |
| libero_object | `lerobot/libero_object_image` | `libero_object_no_noops` |
| libero_goal | `lerobot/libero_goal_image` | `libero_goal_no_noops` |
| libero_10 | `lerobot/libero_10_image` | `libero_10_no_noops` |

## Examples

```
# Finetune pi0 on LIBERO spatial (LeRobot backend)
finetune(policy="pi0", benchmark="libero_spatial", steps=50000)

# Finetune OpenVLA on LIBERO goal (OpenVLA backend, 4 GPUs)
finetune(policy="openvla", benchmark="libero_goal", num_gpus=4, steps=150000)

# Finetune pi0.5 on RoboCasa
finetune(policy="pi0.5", benchmark="robocasa", steps=100000)
```

## Output

Checkpoints saved to `$AGENTROBOT_ROOT/checkpoints/<output_name>/`.
Use the checkpoint path with `run_benchmark` to evaluate:
```
run_benchmark(policy="pi0", benchmark="libero_spatial", checkpoint="checkpoints/pi0-libero_spatial/checkpoint_50000")
```
