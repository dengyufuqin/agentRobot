---
name: deploy_robomimic
description: "Deploy droid_policy_learning model as a WebSocket policy server (local or remote via SSH)"
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
    default: ""
requires:
  bins: [python3]
timeout: 120
command_template: |
  REPO=$AGENTROBOT_ROOT/droid_policy_learning
  mkdir -p $AGENTROBOT_ROOT/logs

  if [ -z "{node}" ] || [ "{node}" = "localhost" ]; then
    LOG=$AGENTROBOT_ROOT/logs/droid_policy_learning-${HOSTNAME:-local}-{port}.log
    echo "Deploying robomimic locally (GPU {gpu_id}, port {port})..."
    if ss -tlnp 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use locally"; exit 1
    fi
    cd $REPO && source .venv/bin/activate
    export PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$REPO
    export CUDA_VISIBLE_DEVICES={gpu_id}
    nohup python3 -m robomimic.scripts.run_policy_server --ckpt {checkpoint} --port {port} --bf16 > $LOG 2>&1 &
    echo "PID=$!"
    echo "Server starting locally on port {port}. Log: $LOG"
  else
    LOG=$AGENTROBOT_ROOT/logs/droid_policy_learning-{node}-{port}.log
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {node} "ss -tlnp | grep :{port}" 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use on {node}"; exit 1
    fi
    ssh -o StrictHostKeyChecking=no {node} "
      cd $REPO && source .venv/bin/activate
      export PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:\$REPO
      export CUDA_VISIBLE_DEVICES={gpu_id}
      nohup python3 -m robomimic.scripts.run_policy_server --ckpt {checkpoint} --port {port} --bf16 > $LOG 2>&1 &
      echo \"PID=\$!\"
    "
    echo "Server starting on {node}:{port} (GPU {gpu_id}). Log: $LOG"
  fi
  echo "Wait ~60s for model to load."
---

# Deploy droid_policy_learning Policy Server

- **Local mode**: Leave `node` empty
- **Remote mode**: Specify `node` for SSH deployment
- Needs ~16GB GPU memory
- Model loads in ~60 seconds
