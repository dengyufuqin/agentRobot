---
name: download_dataset
description: "Download a HuggingFace dataset to a chosen local directory, reporting size and reusing any partial download. Used during finetune onboarding before generating a dataloader."
version: 1.0.0
category: setup
parameters:
  repo_id:
    type: string
    description: "HF dataset repo ID (e.g. 'lerobot/libero_spatial', 'HuggingFaceVLA/libero')"
    required: true
  local_dir:
    type: string
    description: "Absolute target directory; empty = HF cache"
    required: false
  allow_patterns:
    type: string
    description: "Comma-separated glob patterns (e.g. 'data/train/*,meta/*')"
    required: false
requires:
  bins: [python3]
timeout: 3600
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/download_dataset/download_dataset.py \
    --repo-id "{repo_id}" \
    --local-dir "{local_dir}" \
    --allow-patterns "{allow_patterns}"
---

# Download Dataset

Downloads a HuggingFace dataset with progress reporting and resume-on-partial. Intended for the **finetune** onboarding flow: download → validate_dataset → generate_dataloader → validate_dataloader → finetune.

## Exit codes

| Exit | Meaning |
|---|---|
| 0 | Downloaded (prints local path + size GB as JSON on last line) |
| 2 | Repo does not exist |
| 11 | Download started but failed |

## Usage

```
download_dataset(
  repo_id="lerobot/libero_spatial",
  local_dir="/mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/libero_spatial",
)
```
