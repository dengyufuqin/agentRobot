---
name: generate_run_evaluation
description: "Emit a run_evaluation.sh that boots the policy server, waits for its TCP port, runs the benchmark client, and tears down cleanly on exit. The output script is standalone — ready to sbatch or run locally without loading any skill framework."
version: 1.0.0
category: codegen
parameters:
  policy_repo:
    type: string
    required: true
  policy_entry:
    type: string
    description: "Policy entry script path relative to repo (e.g. policy_server.py)"
    required: true
  checkpoint:
    type: string
    required: true
  checkpoint_flag:
    type: string
    default: "--checkpoint"
  policy_port:
    type: integer
    default: 18800
  policy_extra_args:
    type: string
    required: false
  benchmark_repo:
    type: string
    required: true
  benchmark_entry:
    type: string
    description: "Benchmark entry (e.g. run_benchmark.py)"
    required: true
  suite:
    type: string
    default: "libero_spatial"
  task:
    type: string
    default: "0"
  num_trials:
    type: integer
    default: 10
  benchmark_extra_args:
    type: string
    required: false
  boot_timeout:
    type: integer
    default: 180
  out:
    type: string
    description: "Default: {policy_repo}/run_evaluation.sh"
    required: false
requires:
  bins: [python3]
timeout: 30
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/generate_run_evaluation/generate_run_evaluation.py \
    --policy-repo "{policy_repo}" \
    --policy-entry "{policy_entry}" \
    --checkpoint "{checkpoint}" \
    --checkpoint-flag "{checkpoint_flag}" \
    --policy-port {policy_port} \
    --policy-extra-args "{policy_extra_args}" \
    --benchmark-repo "{benchmark_repo}" \
    --benchmark-entry "{benchmark_entry}" \
    --suite "{suite}" \
    --task "{task}" \
    --num-trials {num_trials} \
    --benchmark-extra-args "{benchmark_extra_args}" \
    --boot-timeout {boot_timeout} \
    --out "{out}"
---

# Generate run_evaluation.sh

End-to-end eval launcher: policy server + benchmark client in one script with boot-wait + auto-teardown.

## Exit codes (of the generated script)

| Exit | Meaning |
|---|---|
| 0 | benchmark completed normally |
| 1 | venv missing |
| 4 | policy server died during boot — log tail printed |
| 5 | policy TCP port didn't open within boot_timeout |
| (benchmark's own code) | benchmark process failure |

## Why separate from generate_run_demo

`run_demo.sh` is "just boot the policy and hand me a port". `run_evaluation.sh` adds the benchmark client + orchestration (wait-for-port, teardown). The two compose — demo is useful when debugging the policy side alone.
