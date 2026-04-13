---
name: deploy_openvla
description: Deploy OpenVLA-OFT model as a WebSocket policy server (local or remote via SSH)
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
    description: "HuggingFace checkpoint path"
    default: "moojink/openvla-7b-oft-finetuned-libero-spatial"
requires:
  bins: [python3]
timeout: 120
command_template: |
  REPO=$AGENTROBOT_ROOT/openvla
  LOG=$AGENTROBOT_ROOT/logs/openvla-${HOSTNAME:-local}-{port}.log
  mkdir -p $AGENTROBOT_ROOT/logs

  if [ -z "{node}" ] || [ "{node}" = "localhost" ]; then
    # === Local mode ===
    echo "Deploying OpenVLA locally (GPU {gpu_id}, port {port})..."
    
    # Check if port already in use
    if ss -tlnp 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use locally"
      exit 1
    fi
    
    cd $REPO
    source .venv/bin/activate
    export PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$REPO
    export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
    export CUDA_VISIBLE_DEVICES={gpu_id}
    nohup python3 vla-scripts/policy_server.py \
      --pretrained_checkpoint {checkpoint} \
      --port {port} \
      --execute_steps 1 \
      > $LOG 2>&1 &
    echo "PID=$!"
    echo "Server starting locally on port {port} (GPU {gpu_id}). Log: $LOG"
  else
    # === Remote mode via SSH ===
    LOG=$AGENTROBOT_ROOT/logs/openvla-{node}-{port}.log
    
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {node} "ss -tlnp | grep :{port}" 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use on {node}"
      exit 1
    fi
    
    ssh -o StrictHostKeyChecking=no {node} "
      cd $REPO
      source .venv/bin/activate
      export PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$REPO
      export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
      export CUDA_VISIBLE_DEVICES={gpu_id}
      nohup python3 vla-scripts/policy_server.py \
        --pretrained_checkpoint {checkpoint} \
        --port {port} \
        --execute_steps 1 \
        > $LOG 2>&1 &
      echo \"PID=\$!\"
    "
    echo "Server starting on {node}:{port} (GPU {gpu_id}). Log: $LOG"
  fi
  echo "Wait ~90s for model to load, then test connection."
---

# Deploy OpenVLA Policy Server

Deploys the OpenVLA-OFT 7B vision-language-action model.
- **Local mode**: Leave `node` empty — runs on current machine
- **Remote mode**: Specify `node` (e.g. cn06) — deploys via SSH
- Needs ~16GB GPU memory (1x H100 or similar)
- First run downloads checkpoint (~14GB) from HuggingFace
- Model loads in ~90 seconds
- Inference: ~1.4s per step on H100
