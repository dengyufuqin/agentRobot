---
name: download_model
description: "Download a HuggingFace checkpoint, with *variant awareness*: when the user says 'pi0 on libero' there are usually a base repo and a finetuned repo — picking the wrong one silently wastes hours. This skill lists the candidates, prefers finetuned variants by default, and exits 10 when ambiguous so the agent or user can disambiguate instead of guessing."
version: 1.0.0
category: setup
parameters:
  repo_id:
    type: string
    description: "Exact HF repo ID (e.g. 'lerobot/pi0_libero_finetuned') OR search terms ('pi0 libero')"
    required: true
  local_dir:
    type: string
    description: "Target directory; if empty uses the HF cache"
    required: false
  prefer_finetuned:
    type: string
    description: "'true' to auto-pick the *_finetuned variant when multiple match"
    default: "true"
  allow_patterns:
    type: string
    description: "Comma-separated glob patterns to download (e.g. '*.safetensors,config.json')"
    required: false
  list_only:
    type: string
    description: "'true' to list matching variants without downloading"
    default: "false"
requires:
  bins: [python3]
timeout: 1800
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/download_model/download_model.py \
    --repo-id "{repo_id}" \
    --local-dir "{local_dir}" \
    --prefer-finetuned "{prefer_finetuned}" \
    --allow-patterns "{allow_patterns}" \
    --list-only "{list_only}"
---

# Download Model (variant-aware)

## Why this skill exists

Real failure mode we hit: user says "run pi0 on libero_spatial", agent downloads `lerobot/pi0_libero` (base checkpoint) → benchmark scores 0% → "this algorithm doesn't work." The actual fix was `lerobot/pi0_libero_finetuned` (same repo, task-finetuned).

## Exit codes

| Exit | Meaning |
|---|---|
| 0 | Downloaded exactly one checkpoint |
| 2 | No matching repo on HF |
| 10 | Ambiguous — multiple variants found; JSON report lists `finetuned` vs `base` |
| 11 | Download started but failed mid-stream |

## Usage

```python
# Exact repo → download
download_model(repo_id="lerobot/pi0_libero_finetuned")

# Pattern → preferred variant
download_model(repo_id="pi0 libero")
# → picks lerobot/pi0_libero_finetuned automatically if prefer_finetuned=true

# List-only (no download)
download_model(repo_id="pi05 libero", list_only="true")
# → {"variants": ["lerobot/pi05_libero_finetuned", "lerobot/pi05_libero_finetuned_v044", ...]}
```

On exit-10 the caller should show the candidate list to the user and call again with the exact `repo_id`.
