#!/bin/bash
# Generate a policy_server.py adapter for a repository.
# Arguments: REPO_PATH MODEL_CLASS MODEL_MODULE CHECKPOINT ACTION_DIM FRAMEWORK

REPO_PATH="$1"
MODEL_CLASS="$2"
MODEL_MODULE="$3"
CHECKPOINT="$4"
ACTION_DIM="$5"
FRAMEWORK="$6"

REPO_NAME=$(basename "$REPO_PATH")
TEMPLATE="${AGENTROBOT_ROOT:-.}/agentic/robot_agent/skills/wrap_policy/policy_server_template.py"
OUTPUT="$REPO_PATH/policy_server.py"

if [ ! -d "$REPO_PATH" ]; then
  echo "ERROR: Repository not found at $REPO_PATH"
  exit 1
fi

# Copy template and substitute placeholders
cp "$TEMPLATE" "$OUTPUT"
sed -i "s|{REPO_NAME}|$REPO_NAME|g" "$OUTPUT"
sed -i "s|{MODULE}|$MODEL_MODULE|g" "$OUTPUT"
sed -i "s|{MODEL_CLASS}|$MODEL_CLASS|g" "$OUTPUT"

# Fix f-strings: replace literal REPO_NAME text in f-strings with proper variable
# The template has f"... {REPO_NAME}" which after sed becomes f"... octo"
# This is actually correct for the docstrings/prints.

# Generate config
cat > "$REPO_PATH/_policy_adapter_config.yaml" << EOF
repo_name: "$REPO_NAME"
model_class: "$MODEL_CLASS"
model_module: "$MODEL_MODULE"
checkpoint: "$CHECKPOINT"
action_dim: $ACTION_DIM
framework: "$FRAMEWORK"
EOF

echo "=== Generated policy_server.py ==="
echo "Output: $OUTPUT"
echo ""
echo "--- File content ---"
cat "$OUTPUT"
echo ""
echo "=== Config ==="
cat "$REPO_PATH/_policy_adapter_config.yaml"
echo ""
echo "=== Next Steps ==="
echo "1. Edit $OUTPUT to implement model loading in __init__"
echo "2. Implement observation remapping in _remap_obs"
echo "3. Implement inference in infer()"
echo "4. Test locally: python $OUTPUT --port 18800"
echo "5. Use create_deploy_skill to generate a deployment SKILL.md"
