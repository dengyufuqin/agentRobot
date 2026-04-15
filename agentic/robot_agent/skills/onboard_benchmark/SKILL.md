---
name: onboard_benchmark
description: "Probe a simulator/benchmark repo: verify it imports, enumerate tasks, capture obs/action space. Symmetric to probe_run (which targets policy repos). Covers libero, maniskill, robocasa, calvin, simpler_env out-of-the-box; tries all if benchmark unset."
version: 1.0.0
category: analysis
parameters:
  repo_path:
    type: string
    description: "Absolute path to the benchmark repo (must have a .venv set up)"
    required: true
  venv_python:
    type: string
    description: "Path to venv python3 (default: {repo_path}/.venv/bin/python3)"
    required: false
  benchmark:
    type: string
    description: "Explicit benchmark id: libero, maniskill, robocasa, calvin, simpler. Empty = try all"
    required: false
  timeout:
    type: integer
    description: "Max seconds per probe"
    default: 120
  lite:
    type: string
    description: "If 'true', only verify import + enumerate tasks (no env.reset). Use on login nodes / no-GPU machines. Skips MUJOCO/EGL, runs in <5s per benchmark."
    default: "false"
  out:
    type: string
    description: "Output JSON path"
    required: false
requires:
  bins: [python3]
timeout: 300
command_template: |
  VENV_ARG=""
  if [ -n "{venv_python}" ]; then
    VENV_ARG="--venv-python {venv_python}"
  fi
  BM_ARG=""
  if [ -n "{benchmark}" ]; then
    BM_ARG="--benchmark {benchmark}"
  fi
  LITE_ARG=""
  if [ "{lite}" = "true" ]; then
    LITE_ARG="--lite"
  fi

  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/onboard_benchmark/onboard_benchmark.py \
    --repo-path "{repo_path}" \
    --timeout {timeout} \
    --out "{out}" \
    $VENV_ARG $BM_ARG $LITE_ARG
---

# Onboard Benchmark

Run AFTER `setup_env`. Verifies a simulator can be imported, reset, and stepped — returns a sample task plus obs/action schema the adapter code will need.

| Exit | Meaning |
|---|---|
| 0 | at least one benchmark probe worked — payload in `winner` |
| 2 | import fails / timeout |
| 3 | imports but no factory successfully resets |

## Built-in probes

- `libero` — `libero.libero.benchmark.get_benchmark_dict()` + OffScreenRenderEnv
- `maniskill` — `gymnasium.make("PickCube-v1", obs_mode="rgbd")`
- `robocasa` — `robomimic.envs.env_robosuite.EnvRobosuite`
- `calvin` — `calvin_env.envs.play_table_env.PlayTableSimEnv`
- `simpler` — `simpler_env.make(ENVIRONMENTS[0])`

To add a new benchmark, append an entry to the `PROBES` dict in `onboard_benchmark.py` with a snippet that imports, resets, and assigns `result = {...}`.
