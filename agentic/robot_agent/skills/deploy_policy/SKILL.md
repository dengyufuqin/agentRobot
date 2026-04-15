---
name: deploy_policy
description: "Unified WebSocket policy server deployment. Supersedes the 6 per-repo deploy_* skills (deploy_lerobot, deploy_openvla, deploy_octo, deploy_diffusion_policy, deploy_robomimic, deploy_beso). Runs any repo's policy_server.py locally or on a remote SLURM node with consistent port/log/PID handling."
version: 1.0.0
category: deploy
parameters:
  repo:
    type: string
    description: "Repo directory name under $AGENTROBOT_ROOT (e.g. 'lerobot', 'openvla', 'octo', 'diffusion_policy', 'droid_policy_learning', 'beso')"
    required: true
  checkpoint:
    type: string
    description: "HF model ID or absolute local path"
    required: true
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
    description: "Remote compute node (empty or 'localhost' = local)"
    required: false
  entry_script:
    type: string
    description: "Path to policy_server.py relative to repo (default 'policy_server.py')"
    default: "policy_server.py"
  checkpoint_flag:
    type: string
    description: "CLI flag name for the checkpoint (e.g. '--checkpoint' for lerobot, '--pretrained_checkpoint' for openvla)"
    default: "--checkpoint"
  extra_args:
    type: string
    description: "Extra CLI args appended (e.g. '--device cuda' or '--execute_steps 1')"
    required: false
  pre_activate:
    type: string
    description: "Shell snippet run before launch (e.g. JAX/CUDA path exports for octo). Leave empty for most repos."
    required: false
requires:
  bins: [python3]
timeout: 120
command_template: |
  REPO=$AGENTROBOT_ROOT/{repo}
  VENV=$REPO/.venv/bin/python3
  SCRIPT=$REPO/{entry_script}
  PP=$AGENTROBOT_ROOT/agentic/policy_websocket/src:$REPO
  LOG=$AGENTROBOT_ROOT/logs/{repo}-${HOSTNAME:-local}-{port}.log
  mkdir -p $AGENTROBOT_ROOT/logs

  if [ ! -f "$SCRIPT" ]; then
    echo "ERROR: entry script not found: $SCRIPT"; exit 1
  fi
  if [ ! -x "$VENV" ]; then
    echo "ERROR: venv python not executable: $VENV"; exit 1
  fi

  LAUNCH_CMD="cd $REPO && {pre_activate} export PYTHONPATH=$PP && export CUDA_VISIBLE_DEVICES={gpu_id} && nohup $VENV $SCRIPT {checkpoint_flag} '{checkpoint}' --port {port} {extra_args} > $LOG 2>&1 & echo PID=\$!"

  if [ -z "{node}" ] || [ "{node}" = "localhost" ]; then
    if ss -tlnp 2>/dev/null | grep -q ":{port} "; then
      echo "ERROR: port {port} already in use locally"; exit 1
    fi
    echo "Starting {repo} policy server locally (GPU {gpu_id}, port {port})"
    echo "Checkpoint: {checkpoint}"
    echo "Log: $LOG"
    bash -c "$LAUNCH_CMD"
  else
    LOG=$AGENTROBOT_ROOT/logs/{repo}-{node}-{port}.log
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {node} "ss -tlnp | grep ':{port} '" 2>/dev/null | grep -q :{port}; then
      echo "ERROR: port {port} already in use on {node}"; exit 1
    fi
    echo "Starting {repo} policy server on {node} (GPU {gpu_id}, port {port})"
    ssh -o StrictHostKeyChecking=no {node} "$LAUNCH_CMD"
  fi
  echo "Wait ~60-120s for load, then call test_policy_connection."
---

# Deploy Policy (unified)

Single skill replacing the 6 legacy per-repo deploy skills. All of them were ~95% identical bash — only the repo name and one or two CLI flags differed.

## Why merge

- One code path to maintain for port collision, SSH, logs, PID capture.
- Easier to add features (health probe, retry loop, container mode) once instead of six times.
- The LLM was picking the wrong deploy_* skill anyway — now it just passes `repo=<name>`.

## Migration cheatsheet

| Old call | New call |
|---|---|
| `deploy_lerobot(checkpoint=X)` | `deploy_policy(repo="lerobot", checkpoint=X, extra_args="--device cuda")` |
| `deploy_openvla(checkpoint=X)` | `deploy_policy(repo="openvla", checkpoint=X, entry_script="vla-scripts/policy_server.py", checkpoint_flag="--pretrained_checkpoint", extra_args="--execute_steps 1")` |
| `deploy_octo(checkpoint=X)` | `deploy_policy(repo="octo", checkpoint=X, pre_activate="export PATH=...ptxas...:$PATH &&")` |
| `deploy_diffusion_policy(checkpoint=X)` | `deploy_policy(repo="diffusion_policy", checkpoint=X)` |
| `deploy_robomimic(checkpoint=X)` | `deploy_policy(repo="droid_policy_learning", checkpoint=X)` |
| `deploy_beso(checkpoint=X)` | `deploy_policy(repo="beso", checkpoint=X)` |

## Typical flow

1. `probe_run` — verify the repo's `policy_server.py` actually boots with the checkpoint.
2. `deploy_policy` — start it as a real server.
3. `test_policy_connection` — handshake + 1 inference.
4. `run_benchmark` — evaluate.
