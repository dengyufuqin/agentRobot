---
name: extract_io_spec
description: "Parse probe_run's .probe_io_spec.json and produce a structured obs/action spec summary (image keys, state keys, action dim). This turns the raw forward-call capture into something the LLM can use to write a correct policy_server.py adapter."
version: 1.0.0
category: analysis
parameters:
  spec_file:
    type: string
    description: "Path to the .probe_io_spec.json file produced by probe_run(--io-spec-hook=true)"
    required: true
  out:
    type: string
    description: "Optional path to write the structured JSON summary"
    required: false
requires:
  bins: [python3]
timeout: 30
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/extract_io_spec/extract_io_spec.py \
    --spec-file "{spec_file}" \
    --out "{out}"
---

# Extract IO Spec

Companion to `probe_run`. Reads the raw forward-call capture and emits:

```json
{
  "calls": [ { "module": "PI0Pytorch", "inputs": {...}, "outputs": {...} } ],
  "derived": {
    "image_keys":  ["arg[0].observation.images.front", "arg[0].observation.images.wrist"],
    "state_keys":  ["arg[0].observation.state"],
    "action_shape": [1, 50, 7],
    "action_dim":   7
  }
}
```

## Workflow

1. `probe_run(..., io_spec_hook="true")` → writes `{repo}/.probe_io_spec.json`
2. `extract_io_spec(spec_file=".../.probe_io_spec.json")` → structured derived spec
3. LLM uses `derived.action_dim`, `image_keys`, `state_keys` to write `policy_server.py`

Without this skill the LLM would either read the README (often wrong) or guess.
