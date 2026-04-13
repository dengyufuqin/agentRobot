---
name: validate_policy_server
description: "Validate a generated policy_server.py adapter without loading model weights. Runs syntax check, then import check, then optionally a smoke-test infer() call. Returns a structured pass/fail report with tracebacks so the LLM can iterate on its adapter code without spinning up a real GPU server."
version: 1.0.0
category: meta
parameters:
  repo_path:
    type: string
    description: "Absolute path to the repo containing policy_server.py"
    required: true
  python:
    type: string
    description: "Python interpreter to validate against (e.g. /path/to/repo/.venv/bin/python3). Required because the adapter must import the repo's own model code."
    required: true
  pythonpath:
    type: string
    description: "Colon-separated PYTHONPATH (e.g. '$AGENTROBOT_ROOT/agentic/policy_websocket/src:$AGENTROBOT_ROOT/<repo>'). Defaults to empty."
    default: ""
  mode:
    type: string
    description: "syntax | import | smoke. 'syntax' = py_compile only (instant). 'import' = also import the module (catches missing deps). 'smoke' = also instantiate Policy and call infer({}) with a no-op observation."
    default: "import"
  adapter:
    type: string
    description: "Adapter filename relative to repo_path"
    default: "policy_server.py"
  policy_class:
    type: string
    description: "(smoke mode) Policy class name to instantiate. Auto-detected if omitted."
    default: ""
requires:
  bins: [python3]
timeout: 90
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/validate_policy_server/validate.py "{repo_path}" --python "{python}" --pythonpath "{pythonpath}" --mode "{mode}" --adapter "{adapter}" $( [ -n "{policy_class}" ] && echo --policy-class {policy_class} )
---

# Validate Policy Server

The feedback loop primitive that lets the agent iterate on adapter code without
spinning up a real GPU server.

## Workflow
1. After `wrap_policy` (or after writing an adapter via `write_file`):
   `validate_policy_server(repo_path=..., python=..., mode="import")`
2. If it fails with a missing import: call `fix_deps` or update the adapter
   imports, then retry.
3. Once import succeeds: rerun with `mode="smoke"` to verify the `infer({})`
   contract holds (returns `{"actions": array}`).
4. Only after all three modes pass should the agent start the real server.

## Why no model loading?
Loading real weights takes 60–120s on GPU. We want fast iteration: most adapter
bugs are import errors, missing classes, or wrong return shapes — all of which
can be caught in <5s without touching CUDA.

## How to make adapters validation-friendly
The smoke check passes `checkpoint="__validate_only__"` and tries `_validate_only=True`.
Adapters that want to skip weight loading during validation can detect this:

```python
def __init__(self, **kwargs):
    if kwargs.get("_validate_only"):
        self.model = None  # skip weight loading
        self._action_dim = 7
        return
    # ... real loading code ...
```
