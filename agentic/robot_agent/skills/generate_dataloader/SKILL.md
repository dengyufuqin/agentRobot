---
name: generate_dataloader
description: "Emit a minimal working make_dataloader() factory file for a detected dataset format (lerobot, parquet). Companion to validate_dataset → validate_dataloader. The output is a starter — the LLM is expected to refine it for the specific finetune recipe."
version: 1.0.0
category: setup
parameters:
  format:
    type: string
    description: "Dataset format, typically from validate_dataset (one of: lerobot, parquet)"
    required: true
  repo_id:
    type: string
    description: "HF dataset ID (for lerobot format)"
    required: false
  root:
    type: string
    description: "Local dataset root (for offline use)"
    required: false
  out:
    type: string
    description: "Where to write the factory .py file"
    required: true
requires:
  bins: [python3]
timeout: 30
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/generate_dataloader/generate_dataloader.py \
    --format "{format}" \
    --repo-id "{repo_id}" \
    --root "{root}" \
    --out "{out}"
---

# Generate Dataloader

Emits a starter dataloader factory. Intended workflow:

1. `download_dataset(...)` → local dir
2. `validate_dataset(dataset_dir=...)` → detect format
3. `generate_dataloader(format=..., root=..., out=<repo>/dataloaders/auto_dl.py)`
4. `validate_dataloader(factory_module=<repo>/dataloaders/auto_dl.py)` → verify it yields batches
5. LLM inspects batch keys from step 4 and refines the factory if needed (e.g., add image transforms, rename keys)
6. `finetune(...)`

## Supported formats

| Format | Template uses |
|---|---|
| lerobot | `lerobot.common.datasets.lerobot_dataset.LeRobotDataset` |
| parquet | `pyarrow.parquet` + naive row iterator |

Other formats (WebDataset, Zarr, RLDS) are not yet templated — the LLM should write those by hand using `write_file`.
