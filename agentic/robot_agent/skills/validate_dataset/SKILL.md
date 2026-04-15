---
name: validate_dataset
description: "Inspect a local dataset directory and detect its format (LeRobot v2, WebDataset, Zarr, RLDS, Parquet). Reports episode count, features, and samples so the agent can pick the right dataloader template before generating one."
version: 1.0.0
category: validation
parameters:
  dataset_dir:
    type: string
    description: "Absolute path to the dataset directory"
    required: true
  out:
    type: string
    description: "Optional path to dump the JSON report"
    required: false
requires:
  bins: [python3]
timeout: 60
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/validate_dataset/validate_dataset.py \
    --dataset-dir "{dataset_dir}" \
    --out "{out}"
---

# Validate Dataset

Scans a local dataset dir and identifies the format. Run this after `download_dataset` to decide which dataloader template `generate_dataloader` should emit.

## Exit codes

| Exit | Meaning |
|---|---|
| 0 | Format detected (check `format` field) |
| 2 | Directory does not exist |
| 3 | No known format signature found |

## Report example

```json
{
  "root": "/mnt/.../libero_spatial",
  "format": "lerobot",
  "details": {
    "codebase_version": "v2.1",
    "total_episodes": 432,
    "total_frames": 50132,
    "fps": 20,
    "features": ["observation.images.front", "observation.state", "action", ...]
  }
}
```
