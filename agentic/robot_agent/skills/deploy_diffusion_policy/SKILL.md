---
name: deploy_diffusion_policy
description: "Deploy diffusion_policy model as a WebSocket policy server (local or remote via SSH)"
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
  REPO=$AGENTROBOT_ROOT/diffusion_policy
  mkdir -p $AGENTROBOT_ROOT/logs

  if [ -z "{node}" ] || [ "{node}" = "localhost" ]; then
    LOG=$AGENTROBOT_ROOT/logs/diffusion_policy-${HOSTNAME:-local}-{port}.log
    echo "Deploying diffusion_policy locally (GPU {gpu_id}, port {port})..."
    if ss -tlnp 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use locally"; exit 1
    fi
    cd $REPO && source .venv/bin/activate
    export PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$REPO
    export CUDA_VISIBLE_DEVICES={gpu_id}
    nohup python3 policy_server.py --port {port} --checkpoint {checkpoint} > $LOG 2>&1 &
    echo "PID=$!"
    echo "Server starting locally on port {port}. Log: $LOG"
  else
    LOG=$AGENTROBOT_ROOT/logs/diffusion_policy-{node}-{port}.log
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {node} "ss -tlnp | grep :{port}" 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use on {node}"; exit 1
    fi
    ssh -o StrictHostKeyChecking=no {node} "
      cd $REPO && source .venv/bin/activate
      export PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:\$REPO
      export CUDA_VISIBLE_DEVICES={gpu_id}
      nohup python3 policy_server.py --port {port} --checkpoint {checkpoint} > $LOG 2>&1 &
      echo \"PID=\$!\"
    "
    echo "Server starting on {node}:{port} (GPU {gpu_id}). Log: $LOG"
  fi
  echo "Wait ~120s for model to load."
---

# Deploy diffusion_policy Policy Server

- **Local mode**: Leave `node` empty
- **Remote mode**: Specify `node` for SSH deployment
- Needs ~24GB GPU memory
- Model loads in ~120 seconds
