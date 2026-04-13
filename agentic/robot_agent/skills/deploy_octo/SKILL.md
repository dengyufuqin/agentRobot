---
name: deploy_octo
description: "Deploy octo model as a WebSocket policy server (local or remote via SSH)"
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
    default: "hf://rail-berkeley/octo-small-1.5"
requires:
  bins: [python3]
timeout: 120
command_template: |
  REPO=$AGENTROBOT_ROOT/octo
  mkdir -p $AGENTROBOT_ROOT/logs

  # JAX/CUDA library setup (needed for octo's JAX backend)
  setup_jax_libs() {
    if [ -f "$REPO/.venv/bin/python3" ]; then
      NVIDIA_BASE=$($REPO/.venv/bin/python3 -c 'import nvidia; import os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null)
      if [ -n "$NVIDIA_BASE" ]; then
        export LD_LIBRARY_PATH=${NVIDIA_BASE}/cuda_runtime/lib:${NVIDIA_BASE}/cublas/lib:${NVIDIA_BASE}/cudnn/lib:${NVIDIA_BASE}/cufft/lib:${NVIDIA_BASE}/cusolver/lib:${NVIDIA_BASE}/cusparse/lib:${NVIDIA_BASE}/nvjitlink/lib:${NVIDIA_BASE}/cuda_cupti/lib:${LD_LIBRARY_PATH}
      fi
    fi
    # Add CUDA bin to PATH if available
    for cuda_path in /usr/local/cuda/bin /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin; do
      if [ -d "$cuda_path" ]; then export PATH=$cuda_path:$PATH; break; fi
    done
  }

  if [ -z "{node}" ] || [ "{node}" = "localhost" ]; then
    LOG=$AGENTROBOT_ROOT/logs/octo-${HOSTNAME:-local}-{port}.log
    echo "Deploying octo locally (GPU {gpu_id}, port {port})..."
    if ss -tlnp 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use locally"; exit 1
    fi
    cd $REPO && source .venv/bin/activate
    export PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$REPO
    export CUDA_VISIBLE_DEVICES={gpu_id}
    setup_jax_libs
    nohup python3 policy_server.py --port {port} --checkpoint {checkpoint} > $LOG 2>&1 &
    echo "PID=$!"
    echo "Server starting locally on port {port}. Log: $LOG"
  else
    LOG=$AGENTROBOT_ROOT/logs/octo-{node}-{port}.log
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {node} "ss -tlnp | grep :{port}" 2>/dev/null | grep -q :{port}; then
      echo "ERROR: Port {port} already in use on {node}"; exit 1
    fi
    ssh -o StrictHostKeyChecking=no {node} "
      cd $REPO && source .venv/bin/activate
      export PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:\$REPO
      export CUDA_VISIBLE_DEVICES={gpu_id}
      NVIDIA_BASE=\$(.venv/bin/python3 -c 'import nvidia; import os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null)
      if [ -n \"\$NVIDIA_BASE\" ]; then
        export LD_LIBRARY_PATH=\${NVIDIA_BASE}/cuda_runtime/lib:\${NVIDIA_BASE}/cublas/lib:\${NVIDIA_BASE}/cudnn/lib:\${NVIDIA_BASE}/cufft/lib:\${NVIDIA_BASE}/cusolver/lib:\${NVIDIA_BASE}/cusparse/lib:\${NVIDIA_BASE}/nvjitlink/lib:\${NVIDIA_BASE}/cuda_cupti/lib:\${LD_LIBRARY_PATH}
      fi
      for cuda_path in /usr/local/cuda/bin /opt/nvidia/hpc_sdk/Linux_x86_64/*/cuda/*/bin; do
        [ -d \"\$cuda_path\" ] && export PATH=\$cuda_path:\$PATH && break
      done
      nohup python3 policy_server.py --port {port} --checkpoint {checkpoint} > $LOG 2>&1 &
      echo \"PID=\$!\"
    "
    echo "Server starting on {node}:{port} (GPU {gpu_id}). Log: $LOG"
  fi
  echo "Wait ~60s for model to load."
---

# Deploy octo Policy Server

- **Local mode**: Leave `node` empty
- **Remote mode**: Specify `node` for SSH deployment
- Needs ~8GB GPU memory (JAX backend)
- Model loads in ~60 seconds
