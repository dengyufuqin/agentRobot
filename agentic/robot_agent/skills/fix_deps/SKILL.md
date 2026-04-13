---
name: fix_deps
description: "Diagnose and auto-fix dependency issues in a Python venv. Tests imports, pattern-matches errors against known fixes (missing packages, numpy 2.x, libGL, mujoco version, CUDA mismatch, etc.), applies fixes, and retries until all imports pass."
version: 1.0.0
category: env
parameters:
  repo_path:
    type: string
    description: "Absolute path to the repository with a .venv"
    required: true
  modules:
    type: string
    description: "Comma-separated list of modules to test (auto-detected if omitted)"
    required: false
  max_retries:
    type: integer
    description: "Maximum number of fix-retry rounds"
    default: 5
requires:
  bins: [uv, python3]
timeout: 300
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/fix_deps/fix_deps.py \
    "{repo_path}" \
    --max-retries {max_retries} \
    $([ -n "{modules}" ] && echo "--modules {modules}") \
    --json
---

# Fix Dependencies

Automatically diagnoses and repairs Python dependency issues in a virtual environment.

## How it works

1. **Detect** — auto-discovers importable modules from pyproject.toml/setup.py
2. **Test** — runs `import X` for each module in the venv
3. **Match** — compares errors against 15+ known fix patterns
4. **Fix** — applies the appropriate `uv pip install` command
5. **Retry** — loops until all imports pass or max retries reached

## Known fix patterns

- `ModuleNotFoundError` → installs the missing package (with name mapping, e.g. cv2→opencv-python-headless)
- `libGL.so` missing → switches to opencv-python-headless
- numpy 2.x incompatibility → pins numpy<2
- mujoco/robosuite version mismatch → pins mujoco==2.3.7
- torch.xpu missing (diffusers too new) → pins diffusers<0.27
- cmake required → installs cmake
- CUDA compute capability mismatch → reinstalls torch with cu121

## When to use

- After `setup_env` completes but imports still fail
- When deploying to a new repo and import errors appear
- As a diagnostic tool to understand what's broken in a venv
