---
name: generate_run_demo
description: "Emit a standalone run_demo.sh in the repo that activates its venv, runs the entry script with the checkpoint + port + extra args. The generated script is framework-free — the user can re-run it without loading any agent skills. Pairs with deploy_policy + probe_run + infer_io_spec."
version: 1.0.0
category: codegen
parameters:
  repo_path:
    type: string
    description: "Absolute path to the policy repo"
    required: true
  entry_script:
    type: string
    description: "Entry script path relative to repo (e.g. policy_server.py)"
    required: true
  checkpoint:
    type: string
    description: "Checkpoint path (absolute) or HF repo id"
    required: true
  checkpoint_flag:
    type: string
    description: "CLI flag name for the checkpoint arg"
    default: "--checkpoint"
  port:
    type: integer
    description: "WebSocket port"
    default: 18800
  extra_args:
    type: string
    description: "Extra CLI args appended verbatim"
    required: false
  out:
    type: string
    description: "Output path (default: {repo_path}/run_demo.sh)"
    required: false
requires:
  bins: [python3]
timeout: 30
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/generate_run_demo/generate_run_demo.py \
    --repo-path "{repo_path}" \
    --entry-script "{entry_script}" \
    --checkpoint "{checkpoint}" \
    --checkpoint-flag "{checkpoint_flag}" \
    --port {port} \
    --extra-args "{extra_args}" \
    --out "{out}"
---

# Generate run_demo.sh

One-shot launcher for the agent's current policy/config. The agent uses it to hand the user a standalone, reproducible demo script — which the user can also modify and keep after the session.
