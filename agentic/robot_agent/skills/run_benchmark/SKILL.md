---
name: run_benchmark
description: "Run a complete benchmark evaluation across multiple platforms (LIBERO, ManiSkill, RoboTwin). Automatically starts the policy server, waits for it to load, runs the evaluation client, collects results, and shuts down the server."
version: 2.0.0
category: eval
parameters:
  policy:
    type: string
    description: "Policy name (e.g. openvla, diffusion_policy) — must have a venv and policy_server.py"
    required: true
  checkpoint:
    type: string
    description: "Model checkpoint path or HuggingFace model ID"
    required: true
  benchmark:
    type: string
    description: "Benchmark name. LIBERO: libero_spatial, libero_object, libero_goal, libero_10, libero_90. ManiSkill: maniskill:PickCube-v1, maniskill:StackCube-v1, etc. RoboTwin: robotwin:beat_block_hammer, robotwin:open_laptop, etc. Use <platform>:<task_id> for any task."
    required: true
  num_trials:
    type: integer
    description: "Number of evaluation trials per task"
    default: 5
  port:
    type: integer
    description: "WebSocket port for the policy server"
    default: 18800
  gpu_id:
    type: integer
    description: "GPU device ID (0-7)"
    default: 0
  node:
    type: string
    description: "Compute node (e.g. cn19). With --submit, submits a SLURM job to this node."
    required: false
  server_addr:
    type: string
    description: "Connect to existing server (host:port) instead of starting a new one."
    required: false
  submit:
    type: string
    description: "If 'true', submit as SLURM job (recommended for HPC). Server+eval run together on GPU node."
    default: "true"
requires:
  bins: [python3]
timeout: 120
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/run_benchmark/run_benchmark.py \
    --policy "{policy}" \
    --checkpoint "{checkpoint}" \
    --benchmark "{benchmark}" \
    --num_trials {num_trials} \
    --port {port} \
    --gpu_id {gpu_id} \
    $([ -n "{node}" ] && echo "--node {node}") \
    $([ -n "{server_addr}" ] && echo "--server_addr {server_addr}") \
    $([ "{submit}" = "true" ] && echo "--submit") \
    --log_dir $AGENTROBOT_ROOT/logs/eval_results
---

# Run Benchmark (Multi-Platform)

End-to-end benchmark evaluation in a single command. Supports **LIBERO**, **ManiSkill**, and **RoboTwin** — or any platform with a WebSocket eval script.

## Benchmark Format

```
libero_spatial              → LIBERO spatial suite (10 tasks)
maniskill:PickCube-v1       → ManiSkill PickCube task
robotwin:beat_block_hammer  → RoboTwin dual-arm task
<platform>:<task_id>        → Any task on any registered platform
```

## Three Execution Modes

### Mode 1: SLURM Job (default, recommended for HPC)
```
run_benchmark(policy="openvla", benchmark="libero_spatial", submit="true")
```

### Mode 2: Existing Server
```
run_benchmark(policy="openvla", benchmark="maniskill:PickCube-v1", server_addr="cn06:18800", submit="false")
```

### Mode 3: Local
```
run_benchmark(policy="openvla", benchmark="robotwin:open_laptop", submit="false")
```

## Supported Platforms

### LIBERO (robosuite)
- `libero_spatial` — 10 spatial reasoning tasks (max 220 steps)
- `libero_object` — 10 object manipulation tasks (max 280 steps)
- `libero_goal` — 10 goal-directed tasks (max 300 steps)
- `libero_10` — 10 long-horizon tasks (max 520 steps)
- `libero_90` — 90 diverse tasks (max 400 steps)

### ManiSkill (SAPIEN)
- `maniskill:PickCube-v1` — Pick up a cube
- `maniskill:StackCube-v1` — Stack cubes
- `maniskill:PegInsertionSide-v1` — Peg insertion
- `maniskill:PickSingleYCB-v1` — Pick YCB objects
- `maniskill:PushCube-v1` — Push a cube
- Any ManiSkill env via `maniskill:<env_id>`

### RoboTwin (dual-arm)
- `robotwin:beat_block_hammer` — Hammer block
- `robotwin:handover_block` — Handover task
- `robotwin:stack_blocks_two` — Stack two blocks
- `robotwin:place_bread_basket` — Place bread in basket
- `robotwin:open_laptop` — Open laptop
- Any RoboTwin task via `robotwin:<task_name>`

## Supported Policies

- `openvla` / `openvla-oft` — OpenVLA 7B
- `diffusion_policy` — Diffusion Policy
- Any repo with `policy_server.yaml` is auto-discovered
