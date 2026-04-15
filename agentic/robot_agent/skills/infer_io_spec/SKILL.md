---
name: infer_io_spec
description: "Resolve a repo's obs/action spec from THREE sources in priority order: README scan → probe_run capture → user-provided fallback JSON. Reports conflicts between sources, emits a merged spec + confidence level. Replaces 'read README and guess' with a verified multi-source pipeline."
version: 1.0.0
category: analysis
parameters:
  repo_path:
    type: string
    description: "Absolute path to the repo"
    required: true
  probe_spec_file:
    type: string
    description: "Path to probe_run's .probe_io_spec.json (default: {repo_path}/.probe_io_spec.json)"
    required: false
  user_fallback:
    type: string
    description: "Optional path to a JSON file with manually-specified {action_dim, image_shape, state_dim} — used when README + probe don't agree or both fail"
    required: false
  out:
    type: string
    description: "Where to write the merged spec JSON"
    required: false
requires:
  bins: [python3]
timeout: 60
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/infer_io_spec/infer_io_spec.py \
    --repo-path "{repo_path}" \
    --probe-spec-file "{probe_spec_file}" \
    --user-fallback "{user_fallback}" \
    --out "{out}"
---

# Infer IO Spec (multi-source)

The policy_server.py adapter needs ground-truth obs/action dims. This skill combines three sources and reports confidence:

| Source | Priority | When used |
|---|---|---|
| `user` | highest | User passed a fallback JSON (human-verified) |
| `probe` | high | `probe_run --io-spec-hook` captured real tensor shapes |
| `readme` | low | Regex scan of README for `action_dim`, image shape patterns |

## Exit codes

| Exit | Meaning |
|---|---|
| 0 | merged spec produced (may include `conflicts` warnings) |
| 1 | repo not found |
| 3 | LOW confidence — caller should ask user for fallback |

## Output

```json
{
  "action_dim": 7,
  "image_shape": [3, 224, 224],
  "state_dim": 8,
  "sources_used": ["action_dim←probe", "image_shape←probe", "state_dim←readme"],
  "conflicts": [],
  "confidence": "high",
  "per_source": [...]
}
```

## Why this exists

- README says "7-DOF" but `policy_server.py` really needs `action_dim=7` scoped to the *flat* output. Often the readme says 8 (joints) when the action is 7 (end-effector). Probe catches this.
- probe captures SiglipVisionModel (encoder submodule), not the top-level policy, in some architectures — user fallback can override.
- Default flow: call `probe_run(io_spec_hook="true")` → `infer_io_spec(repo_path=...)`. If EXIT 3, prompt user for a fallback JSON then re-run.
