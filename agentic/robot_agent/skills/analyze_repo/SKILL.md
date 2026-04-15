---
name: analyze_repo
description: "Analyze a repo's code structure for robot policy integration. Accepts a GitHub URL (clones it) OR a local path / file:// URL (skips clone). Finds model loading, inference entry points, observation/action spaces, and dependencies."
version: 1.0.0
category: meta
parameters:
  repo_url:
    type: string
    description: "GitHub repository URL (e.g. https://github.com/moojink/openvla-oft)"
    required: true
  branch:
    type: string
    description: "Branch to clone"
    default: "main"
  target_dir:
    type: string
    description: "Directory to clone into (default: $AGENTROBOT_ROOT/<repo_name>)"
    required: false
requires:
  bins: [git, python3]
timeout: 300
command_template: |
  REPO_URL="{repo_url}"
  BRANCH="{branch}"
  BASE_DIR=${AGENTROBOT_ROOT:-.}

  # Detect local path: strip file:// prefix, or accept absolute/relative paths
  LOCAL_PATH="${REPO_URL#file://}"
  if [ -d "$LOCAL_PATH" ]; then
    TARGET="$LOCAL_PATH"
    REPO_NAME=$(basename "$TARGET")
    echo "=== Step 1: Local repo detected — skipping clone ==="
    echo "Using existing directory: $TARGET"
  else
    # Extract repo name from URL
    REPO_NAME=$(basename "$REPO_URL" .git)
    TARGET="{target_dir}"
    if [ -z "$TARGET" ]; then
      TARGET="$BASE_DIR/$REPO_NAME"
    fi

    echo "=== Step 1: Clone Repository ==="
    if [ -d "$TARGET" ]; then
      echo "Repository already exists at $TARGET, pulling latest..."
      cd "$TARGET" && git pull 2>&1 | tail -3
    else
      git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$TARGET" 2>&1
    fi
  fi
  
  echo ""
  echo "=== Step 2: Project Structure ==="
  cd "$TARGET"
  echo "Root files:"
  ls -la *.py *.toml *.cfg *.txt *.yaml *.yml 2>/dev/null | head -20
  echo ""
  echo "Key directories:"
  find . -maxdepth 2 -type d -not -path './.git*' -not -path './.venv*' -not -path './node_modules*' | sort | head -30
  
  echo ""
  echo "=== Step 3: Dependencies ==="
  if [ -f "pyproject.toml" ]; then
    echo "[pyproject.toml]"
    python3 -c "
  import tomllib
  with open('pyproject.toml','rb') as f: d=tomllib.load(f)
  deps = d.get('project',{}).get('dependencies',[])
  print('Dependencies:', deps[:20])
  extras = d.get('project',{}).get('optional-dependencies',{})
  for k,v in list(extras.items())[:3]: print(f'  [{k}]: {v[:5]}')
  " 2>/dev/null || cat pyproject.toml | head -40
  fi
  if [ -f "requirements.txt" ]; then
    echo "[requirements.txt]"
    head -20 requirements.txt
  fi
  if [ -f "setup.py" ]; then
    echo "[setup.py found]"
    grep -E "install_requires|name=" setup.py | head -10
  fi
  
  echo ""
  echo "=== Step 4: Model & Inference Analysis ==="
  echo "--- Python files with 'model' or 'policy' in name ---"
  find . -name "*.py" -not -path "./.venv/*" | xargs grep -l -i "class.*policy\|class.*model\|def.*infer\|def.*predict\|def.*forward" 2>/dev/null | head -15
  
  echo ""
  echo "--- Entry points (main, argparse, click) ---"
  grep -rl "if __name__\|argparse\|@click" --include="*.py" . 2>/dev/null | grep -v ".venv" | head -10
  
  echo ""
  echo "--- Observation/Action patterns ---"
  grep -rn "obs\|observation\|action_dim\|action_space\|predict\|infer" --include="*.py" . 2>/dev/null | grep -v ".venv" | grep -iv "import\|#" | head -20
  
  echo ""
  echo "--- Framework detection ---"
  for fw in torch jax tensorflow flax transformers diffusers gymnasium robosuite; do
    if grep -rq "$fw" --include="*.py" --include="*.toml" --include="*.txt" . 2>/dev/null; then
      echo "  Found: $fw"
    fi
  done
  
  echo ""
  echo "--- Existing WebSocket/server patterns ---"
  grep -rl "websocket\|WebSocket\|serve\|server\|flask\|fastapi\|grpc" --include="*.py" . 2>/dev/null | grep -v ".venv" | head -10
  
  echo ""
  echo "=== Step 5: Key File Contents (heads) ==="
  # Show head of most likely inference files
  for f in $(find . -name "*.py" -not -path "./.venv/*" | xargs grep -l "def.*infer\|def.*predict\|def.*forward\|class.*Policy" 2>/dev/null | head -3); do
    echo "--- $f (first 60 lines) ---"
    head -60 "$f"
    echo "..."
  done
  
  echo ""
  echo "=== Analysis Complete ==="
  echo "REPO_PATH=$TARGET"
  echo "REPO_NAME=$REPO_NAME"
---

# Analyze Repository

Meta-skill that clones and deeply analyzes a GitHub repository to understand:
1. **Project structure** — directories, config files, entry points
2. **Dependencies** — Python packages, framework (PyTorch/JAX/etc.)
3. **Model architecture** — how the model is loaded and runs inference
4. **Observation/Action space** — input/output format for policy integration
5. **Existing server patterns** — any WebSocket/HTTP server already present

Use this as the first step before `wrap_policy` and `create_deploy_skill`.
The output gives the LLM enough context to generate a policy_websocket adapter.
