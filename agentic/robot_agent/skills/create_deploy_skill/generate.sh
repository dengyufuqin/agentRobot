#!/bin/bash
# Generate a deploy SKILL.md for a new repository.
# Arguments: REPO_PATH SKILL_NAME CHECKPOINT GPU_MEM LOAD_TIME ACTION_DIM ENV_SETUP

REPO_PATH="$1"
SKILL_NAME="$2"
CHECKPOINT="$3"
GPU_MEM="$4"
LOAD_TIME="$5"
ACTION_DIM="$6"
ENV_SETUP="$7"

REPO_NAME=$(basename "$REPO_PATH")
SKILLS_DIR=${AGENTROBOT_ROOT:-.}/agentic/robot_agent/skills
SKILL_DIR="$SKILLS_DIR/$SKILL_NAME"

# Check policy_server.py exists
if [ ! -f "$REPO_PATH/policy_server.py" ]; then
  echo "ERROR: No policy_server.py found in $REPO_PATH"
  echo "Run wrap_policy first to generate the adapter."
  exit 1
fi

# Create skill directory
mkdir -p "$SKILL_DIR"

# Generate the deploy SKILL.md with local+remote support
cat > "$SKILL_DIR/SKILL.md" << SKILLEOF
---
name: $SKILL_NAME
description: "Deploy $REPO_NAME model as a WebSocket policy server (local or remote via SSH)"
version: 2.0.0
category: deploy
parameters:
  node:
    type: string
    description: "Compute node to deploy on (e.g. cn06). Leave empty for local deployment."
    required: false
  port:
    type: integer
    description: "Port for the WebSocket policy server"
    default: 18800
  gpu_id:
    type: integer
    description: "GPU device ID to use"
    default: 0
  checkpoint:
    type: string
    description: "Model checkpoint path"
    default: "$CHECKPOINT"
requires:
  bins: [python3]
timeout: 120
command_template: |
  REPO=$REPO_PATH
  mkdir -p \$AGENTROBOT_ROOT/logs

  if [ -z "{node}" ] || [ "{node}" = "localhost" ]; then
    LOG=\$AGENTROBOT_ROOT/logs/$REPO_NAME-\${HOSTNAME:-local}-{port}.log
    echo "Deploying $REPO_NAME locally (GPU {gpu_id}, port {port})..."
    if ss -tlnp 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use locally"; exit 1
    fi
    cd \$REPO && $ENV_SETUP
    export PYTHONPATH=\$AGENTROBOT_ROOT/agentic/policy_websocket/src:\$REPO
    export CUDA_VISIBLE_DEVICES={gpu_id}
    nohup python3 policy_server.py --port {port} --checkpoint {checkpoint} > \$LOG 2>&1 &
    echo "PID=\$!"
    echo "Server starting locally on port {port}. Log: \$LOG"
  else
    LOG=\$AGENTROBOT_ROOT/logs/$REPO_NAME-{node}-{port}.log
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {node} "ss -tlnp | grep :{port}" 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use on {node}"; exit 1
    fi
    ssh -o StrictHostKeyChecking=no {node} "
      cd \\\$REPO && $ENV_SETUP
      export PYTHONPATH=\$AGENTROBOT_ROOT/agentic/policy_websocket/src:\\\$REPO
      export CUDA_VISIBLE_DEVICES={gpu_id}
      nohup python3 policy_server.py --port {port} --checkpoint {checkpoint} > \$LOG 2>&1 &
      echo \\\"PID=\\\\\\\$!\\\"
    "
    echo "Server starting on {node}:{port} (GPU {gpu_id}). Log: \$LOG"
  fi
  echo "Wait ~${LOAD_TIME}s for model to load."
---

# Deploy $REPO_NAME Policy Server

- **Local mode**: Leave \`node\` empty
- **Remote mode**: Specify \`node\` for SSH deployment
- Needs ~${GPU_MEM}GB GPU memory
- Model loads in ~${LOAD_TIME} seconds
- Output: ${ACTION_DIM}D action
SKILLEOF

echo "=== Skill Created ==="
echo "Directory: $SKILL_DIR"
echo ""
echo "--- Generated SKILL.md ---"
cat "$SKILL_DIR/SKILL.md"
echo ""
echo "=== Done ==="
echo "New skill '$SKILL_NAME' will be auto-loaded on next agent run."
