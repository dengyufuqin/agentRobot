---
name: check_finetune_capability
description: "Scan a repo for existing train/finetune scripts BEFORE writing new ones. Detects train*.py / finetune*.py files, README 'Training' sections + CLI samples, pyproject console scripts. Emits a recommendation so the agent reuses existing infrastructure instead of hand-rolling a training loop."
version: 1.0.0
category: analysis
parameters:
  repo_path:
    type: string
    description: "Absolute path to the repo"
    required: true
  out:
    type: string
    description: "Optional path to write the JSON report"
    required: false
requires:
  bins: [python3]
timeout: 60
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/check_finetune_capability/check_finetune_capability.py \
    --repo-path "{repo_path}" \
    --out "{out}"
---

# Check Finetune Capability

Decide whether to write a new training loop or reuse what's already in the repo.

| Exit | Meaning |
|---|---|
| 0 | existing train/finetune scripts found — **do not** rewrite |
| 3 | nothing found — agent may need to write one, or this repo isn't trainable |

## Detection sources

1. **File names**: `train.py`, `finetune.py`, `scripts/train_*.py`, `tools/finetune_*.py`
2. **README**: `## Training`, `## Fine-tuning` headings + code-fence `python train.py ...` samples
3. **pyproject.toml**: `[project.scripts]` entries like `lerobot-train = lerobot.scripts.train:main`

## Why this exists

Agents love writing "a quick training loop" in a new file. That usually costs hours and produces a buggy reimplementation. 90% of robotics repos ship a training entrypoint — use it.
