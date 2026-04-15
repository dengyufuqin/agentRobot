---
name: probe_run
description: "Run a repo's own demo/deploy/inference script until it hits a success marker (e.g. 'server ready', 'step: 1', 'model loaded'), then kill it cleanly. Used to verify the repo works with its checkpoint BEFORE writing a policy_server.py adapter — and to optionally capture actual tensor shapes so obs/action spec is seen, not guessed."
version: 1.0.0
category: analysis
parameters:
  repo_path:
    type: string
    description: "Absolute path to the repository"
    required: true
  entry_script:
    type: string
    description: "Path to the demo/deploy/inference script to run, relative to repo_path (e.g. 'scripts/demo.py')"
    required: true
  venv_python:
    type: string
    description: "Absolute path to the venv's python3. Defaults to {repo_path}/.venv/bin/python3"
    required: false
  timeout:
    type: integer
    description: "Max seconds to wait for a success marker before giving up"
    default: 500
  success_markers:
    type: string
    description: "Pipe-separated regex patterns. If empty, uses sensible defaults (training step 1/10/100, server ready, model loaded, Uvicorn running, etc.)"
    required: false
  extra_args:
    type: string
    description: "Space-separated CLI args forwarded to the entry script (e.g. '--checkpoint foo --port 18800')"
    required: false
  io_spec_hook:
    type: string
    description: "If 'true', monkey-patches torch.nn.Module.__call__ AND (optionally) the target class's .forward to capture the first forward call's tensor shapes into {repo_path}/.probe_io_spec.json."
    default: "false"
  target_classes:
    type: string
    description: "Comma-separated class-name substrings whose .forward() should be patched (e.g. 'PI0Pytorch,OpenVLAForAction'). REQUIRED when the policy class calls self.forward() directly instead of self(x) — typical for pi0/VLA architectures. Without this, the hook only catches the first submodule called via __call__ (usually a vision encoder, which gives wrong action_dim)."
    required: false
requires:
  bins: [python3]
timeout: 600
command_template: |
  SCRIPT_DIR=$(dirname "$0")
  PROBE_PY="$AGENTROBOT_ROOT/agentic/robot_agent/skills/probe_run/probe_run.py"

  VENV_ARG=""
  if [ -n "{venv_python}" ]; then
    VENV_ARG="--venv-python {venv_python}"
  fi
  MARKERS_ARG=""
  if [ -n "{success_markers}" ]; then
    MARKERS_ARG="--success-markers {success_markers}"
  fi
  EXTRA_ARG=""
  if [ -n "{extra_args}" ]; then
    EXTRA_ARG="--extra-args {extra_args}"
  fi
  TARGETS_ARG=""
  if [ -n "{target_classes}" ]; then
    TARGETS_ARG="--target-classes {target_classes}"
  fi

  python3 "$PROBE_PY" \
    --repo-path "{repo_path}" \
    --entry-script "{entry_script}" \
    --timeout {timeout} \
    --io-spec-hook {io_spec_hook} \
    $VENV_ARG $MARKERS_ARG $EXTRA_ARG $TARGETS_ARG
---

# Probe-run

Runs a repository's own entry script (demo, deploy server, inference example, training launcher) until it reaches a **success marker**, then kills it cleanly. This replaces "reading the README and guessing" with "running the code and watching what it does."

## When to use

- **Onboarding a new repo**: before writing a `policy_server.py` adapter, confirm the repo's own demo actually loads the checkpoint and produces output.
- **Verifying a checkpoint**: confirm a just-downloaded checkpoint is compatible with the repo's code (not base-vs-finetuned mismatch, not shape mismatch).
- **Extracting obs/action spec**: with `io_spec_hook=true`, capture the first forward pass's input/output tensor shapes — much more reliable than inferring from documentation.

## Success / failure semantics

| Exit code | Meaning |
|---|---|
| 0 | success marker seen, process killed cleanly |
| 2 | timeout with no marker — code ran but didn't reach expected checkpoint |
| 3 | process exited 0 but no marker — script exits too early, marker list is wrong |
| 4 | error pattern detected (traceback / OOM / import error) |
| ≥5 | process failed on its own with that exit code |

## Example

```
probe_run(
    repo_path="/path/to/new_repo",
    entry_script="scripts/demo.py",
    extra_args="--checkpoint /path/to/model.pt",
    timeout=300,
    io_spec_hook="true",
)
```

If it exits 0: read `{repo_path}/.probe_io_spec.json` for the actual tensor shapes and proceed to write the adapter.
If it exits 4: read the traceback — dependency missing, checkpoint incompatible, or code bug.
