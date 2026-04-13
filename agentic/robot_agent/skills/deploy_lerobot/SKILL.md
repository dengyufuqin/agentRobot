---
name: deploy_lerobot
description: "Deploy LeRobot models (pi0, pi0.5, SmolVLA, xVLA) as a WebSocket policy server. Supports any LeRobot-format HuggingFace checkpoint."
version: 1.0.0
category: deploy
parameters:
  checkpoint:
    type: string
    description: "HuggingFace model ID or local path (e.g. lerobot/pi05_libero_finetuned_v044, HuggingFaceVLA/smolvla_libero)"
    default: "lerobot/pi05_libero_finetuned_v044"
  port:
    type: integer
    description: "WebSocket server port"
    default: 18800
  gpu_id:
    type: integer
    description: "GPU device ID"
    default: 0
  node:
    type: string
    description: "Compute node for remote deployment (leave empty for local)"
    required: false
requires:
  bins: [python3]
timeout: 120
command_template: |
  REPO=$AGENTROBOT_ROOT/lerobot
  VENV=$REPO/.venv/bin/python3
  SCRIPT=$REPO/policy_server.py
  PP=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$REPO

  if [ -z "{node}" ] || [ "{node}" = "localhost" ]; then
    # === Local mode ===
    echo "Starting LeRobot server locally (GPU {gpu_id}, port {port})..."
    echo "Checkpoint: {checkpoint}"
    CUDA_VISIBLE_DEVICES={gpu_id} PYTHONPATH=$PP $VENV $SCRIPT \
      --checkpoint "{checkpoint}" --port {port} --device cuda &
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    sleep 5
    for i in $(seq 1 60); do
      curl -s http://localhost:{port}/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
      kill -0 $SERVER_PID 2>/dev/null || { echo "Server died"; exit 1; }
      sleep 10
    done
    echo "LeRobot server running on localhost:{port} (PID $SERVER_PID)"
  else
    # === Remote mode via SSH ===
    echo "Starting LeRobot server on {node} (GPU {gpu_id}, port {port})..."
    ssh {node} "cd $REPO && CUDA_VISIBLE_DEVICES={gpu_id} PYTHONPATH=$PP $VENV $SCRIPT \
      --checkpoint '{checkpoint}' --port {port} --device cuda" &
    echo "Server starting on {node}:{port}"
    sleep 10
    for i in $(seq 1 60); do
      curl -s http://{node}:{port}/healthz > /dev/null 2>&1 && { echo "Server ready after ${i}0s"; break; }
      sleep 10
    done
    echo "LeRobot server running on {node}:{port}"
  fi
---

# Deploy LeRobot

Deploy any LeRobot-format model as a WebSocket policy server. Auto-detects model type (pi0, pi0.5, SmolVLA, xVLA) from the checkpoint config.

## Example: Local deployment
```
deploy_lerobot(checkpoint="lerobot/pi05_libero_finetuned_v044", port=18800, gpu_id=0)
```

## Example: Remote deployment
```
deploy_lerobot(checkpoint="HuggingFaceVLA/smolvla_libero", port=18800, gpu_id=0, node="cn19")
```

## Compatible checkpoints (HuggingFace)
- `lerobot/pi05_libero_finetuned_v044` — pi0.5 (4.1B params)
- `HuggingFaceVLA/smolvla_libero` — SmolVLA
- `lerobot/pi0_libero_finetuned_v044` — pi0
- Any LeRobot-format checkpoint with `config.json` containing `type` field
