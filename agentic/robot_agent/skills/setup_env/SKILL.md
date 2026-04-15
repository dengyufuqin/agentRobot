---
name: setup_env
description: "Set up a Python virtual environment for a repository using uv. Installs all dependencies from pyproject.toml or requirements.txt. Also installs policy_websocket as an editable dependency."
version: 1.0.0
category: env
parameters:
  repo_path:
    type: string
    description: "Absolute path to the repository"
    required: true
  python_version:
    type: string
    description: "Python version to use (e.g. 3.11)"
    default: "3.11"
  extra_deps:
    type: string
    description: "Additional pip packages to install (space-separated)"
    required: false
  no_deps_install:
    type: string
    description: "If 'true', install the repo package with --no-deps (useful when version conflicts exist)"
    default: "false"
  smoke:
    type: string
    description: "Comma-separated module names to import-test after install (e.g. 'torch,transformers,lerobot'). Non-zero exit if any fail — catches broken deps before the agent wastes time trying to run the repo."
    required: false
requires:
  bins: [uv]
timeout: 600
command_template: |
  REPO="{repo_path}"
  PY_VER="{python_version}"
  POLICY_WS=$AGENTROBOT_ROOT/agentic/policy_websocket

  if [ ! -d "$REPO" ]; then
    echo "ERROR: Repo not found at $REPO"
    exit 1
  fi

  cd "$REPO"
  echo "=== Setting up environment for $(basename $REPO) ==="

  # Create venv if not exists
  if [ ! -d ".venv" ]; then
    echo "Creating venv with Python $PY_VER..."
    uv venv --python "$PY_VER" 2>&1 | tail -3
  else
    echo "Venv already exists at .venv/"
  fi

  # Detect CUDA version and pin compatible torch
  CUDA_VER=""
  if command -v nvidia-smi &>/dev/null; then
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
  fi
  # If no GPU locally, check for HPC SDK or known cluster setup
  if [ -z "$CUDA_VER" ]; then
    if [ -d /opt/nvidia/hpc_sdk ]; then
      CUDA_VER="570"  # HPC cluster with NVIDIA SDK: assuming CUDA 12.x
      echo "No GPU on this node, but HPC SDK found — assuming CUDA 12 cluster"
    fi
  fi
  if [ -n "$CUDA_VER" ]; then
    echo "Detected NVIDIA driver major version: $CUDA_VER"
    # Pin torch to cu121 if driver doesn't support CUDA 13
    # Driver 570.x supports CUDA 12.8, not CUDA 13
    if [ "$CUDA_VER" -lt 600 ] 2>/dev/null; then
      echo "Pinning torch to cu121 (cluster CUDA 12.x)"
      uv pip install --python .venv/bin/python3 "torch>=2.2,<2.8" "torchvision" \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple/ \
        --index-strategy unsafe-best-match 2>&1 | tail -5
    fi
  fi

  # Install dependencies
  if [ -f "pyproject.toml" ]; then
    echo ""
    echo "Installing from pyproject.toml..."
    if [ "{no_deps_install}" = "true" ]; then
      uv pip install -e . --no-deps 2>&1 | tail -5
    else
      uv pip install -e "." 2>&1 | tail -10
    fi
  elif [ -f "requirements.txt" ]; then
    echo ""
    echo "Installing from requirements.txt..."
    uv pip install -r requirements.txt 2>&1 | tail -10
  elif [ -f "setup.py" ]; then
    echo ""
    echo "Installing from setup.py..."
    uv pip install -e . 2>&1 | tail -10
  else
    echo "WARNING: No dependency file found"
  fi

  # Install policy_websocket
  echo ""
  echo "Installing policy_websocket..."
  uv pip install -e "$POLICY_WS" --no-deps 2>&1 | tail -3

  # Install extra dependencies
  if [ -n "{extra_deps}" ]; then
    echo ""
    echo "Installing extra deps: {extra_deps}"
    uv pip install {extra_deps} 2>&1 | tail -5
  fi

  # Verify
  echo ""
  echo "=== Verification ==="
  .venv/bin/python3 --version
  .venv/bin/python3 -c "import policy_websocket; print('policy_websocket: OK')" 2>&1
  
  # Check key frameworks
  for pkg in torch jax tensorflow transformers; do
    .venv/bin/python3 -c "import $pkg; print(f'$pkg: {getattr($pkg, \"__version__\", \"OK\")}')" 2>/dev/null
  done

  # Auto-diagnose and fix common dependency issues
  FIX_DEPS=$AGENTROBOT_ROOT/agentic/robot_agent/skills/fix_deps/fix_deps.py
  if [ -f "$FIX_DEPS" ]; then
    echo ""
    echo "=== Auto-fixing dependencies ==="
    python3 "$FIX_DEPS" "$REPO" --max-retries 3 2>&1
  fi

  # Smoke test: import the modules the user specified
  if [ -n "{smoke}" ]; then
    echo ""
    echo "=== Smoke import test ==="
    SMOKE_FAIL=0
    for mod in $(echo "{smoke}" | tr ',' ' '); do
      if .venv/bin/python3 -c "import $mod" 2>&1 | head -20; then
        :  # success — no output from python if import ok
      else
        :
      fi
      if .venv/bin/python3 -c "import $mod" 2>/dev/null; then
        echo "  ✓ $mod"
      else
        echo "  ✗ $mod — IMPORT FAILED"
        .venv/bin/python3 -c "import $mod" 2>&1 | tail -5
        SMOKE_FAIL=1
      fi
    done
    if [ "$SMOKE_FAIL" = "1" ]; then
      echo "=== Smoke test FAILED — some imports broken ==="
      exit 2
    fi
    echo "=== Smoke test passed ==="
  fi

  echo ""
  echo "=== Environment ready at $REPO/.venv ==="
---

# Setup Environment

Creates a Python virtual environment using `uv` and installs all dependencies.
Supports repos with `pyproject.toml`, `requirements.txt`, or `setup.py`.
Always installs `policy_websocket` as an editable dependency for policy server integration.

Use `no_deps_install=true` when the repo has strict version pinning that conflicts with other packages.
