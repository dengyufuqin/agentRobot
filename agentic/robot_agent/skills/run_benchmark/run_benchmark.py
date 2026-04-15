#!/usr/bin/env python3
"""
Run a benchmark evaluation end-to-end:
  1. Start policy server on a GPU node via SLURM or locally
  2. Wait for server to be ready (healthcheck)
  3. Run evaluation client (LIBERO / ManiSkill / RoboTwin / custom)
  4. Kill server, report results

Supports pluggable eval clients — any benchmark with a WebSocket eval script works.

Benchmark format:
    libero_spatial              → LIBERO spatial suite
    maniskill:PickCube-v1       → ManiSkill PickCube task
    robotwin:beat_block_hammer  → RoboTwin task

Usage (one-sentence deployment — checkpoint auto-resolved from registry):
    python run_benchmark.py --policy openvla --benchmark libero_spatial
    python run_benchmark.py --policy pi0.5 --benchmark libero_10
    python run_benchmark.py --policy pi0 --benchmark libero_goal --num_trials 10

  With explicit checkpoint (overrides registry):
    python run_benchmark.py --policy openvla --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
        --benchmark libero_spatial --num_trials 5
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

AGENT_ROOT = Path(os.environ.get("AGENTROBOT_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent.parent)))

# Add policy_websocket to path for eval_registry imports
_PW_SRC = str(AGENT_ROOT / "agentic" / "policy_websocket" / "src")
if _PW_SRC not in sys.path:
    sys.path.insert(0, _PW_SRC)

# Nodes with known EGL rendering issues (SIGABRT on MuJoCo/robosuite)
EGL_EXCLUDE_NODES = os.environ.get("EGL_EXCLUDE_NODES", "cn19,cn23")

# ---------------------------------------------------------------------------
# Eval clients — each defines how to run evaluation for a benchmark platform
# ---------------------------------------------------------------------------
EVAL_CLIENTS = {
    "libero": {
        "eval_python": str(AGENT_ROOT / "LIBERO" / ".venv" / "bin" / "python3"),
        "eval_script": str(AGENT_ROOT / "LIBERO" / "scripts" / "run_eval.py"),
        "eval_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "LIBERO"),
        ],
        "eval_args": [
            "--policy_server_addr", "{server_addr}",
            "--policy", "{policy_flag}",
            "--task_suite_name", "{task_id}",
            "--num_trials_per_task", "{num_trials}",
            "--arm_controller", "{arm_controller}",
            "--log_dir", "{log_dir}",
            "--no_save_video",
        ],
        "env_vars": {"MUJOCO_GL": "egl"},
    },
    "maniskill": {
        "eval_python": str(AGENT_ROOT / "ManiSkill" / ".venv" / "bin" / "python3"),
        "eval_script": str(AGENT_ROOT / "ManiSkill" / "scripts" / "run_eval.py"),
        "eval_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "ManiSkill"),
        ],
        "eval_args": [
            "--policy_server_addr", "{server_addr}",
            "--policy", "{policy_flag}",
            "--env_id", "{task_id}",
            "--num_trials", "{num_trials}",
            "--log_dir", "{log_dir}",
            "--no_save_video",
        ],
        "env_vars": {},
    },
    "robotwin": {
        "eval_python": str(AGENT_ROOT / "RoboTwin" / ".venv" / "bin" / "python3"),
        "eval_script": str(AGENT_ROOT / "RoboTwin" / "script" / "run_eval_ws.py"),
        "eval_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "RoboTwin"),
        ],
        "eval_args": [
            "--policy_server_addr", "{server_addr}",
            "--policy", "{policy_flag}",
            "--task_name", "{task_id}",
            "--action_type", "{action_type}",
            "--num_trials", "{num_trials}",
            "--log_dir", "{log_dir}",
            "--no_save_video",
        ],
        "env_vars": {},
    },
    "robocasa": {
        "eval_python": str(AGENT_ROOT / "robocasa" / ".venv" / "bin" / "python3"),
        "eval_script": str(AGENT_ROOT / "robocasa" / "scripts" / "run_eval.py"),
        "eval_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "robocasa"),
        ],
        "eval_args": [
            "--policy_server_addr", "{server_addr}",
            "--policy", "{policy_flag}",
            "--task_name", "{task_id}",
            "--num_trials", "{num_trials}",
            "--arm_controller", "{arm_controller}",
            "--log_dir", "{log_dir}",
            "--no_save_video",
        ],
        "env_vars": {"MUJOCO_GL": "egl"},
    },
}

# ---------------------------------------------------------------------------
# Known policies: name → how to start the server
# ---------------------------------------------------------------------------
POLICY_CONFIGS = {
    "openvla": {
        "server_python": str(AGENT_ROOT / "openvla" / ".venv" / "bin" / "python3"),
        "server_script": str(AGENT_ROOT / "openvla" / "vla-scripts" / "policy_server.py"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "openvla"),
        ],
        "server_args": ["--pretrained_checkpoint", "{checkpoint}", "--port", "{port}", "--execute_steps", "1"],
        "env_vars": {"HF_HOME": os.path.expanduser("~/.cache/huggingface")},
        "arm_controller": "cartesian_pose",
        "policy_flag": "openvla-oft",
    },
    "openvla-oft": {"alias": "openvla"},
    # LeRobot-based model aliases — all use the same server infrastructure
    "pi0": {"alias": "lerobot"},
    "pi0.5": {"alias": "lerobot"},
    "smolvla": {"alias": "lerobot"},
    "lerobot": {
        "server_python": str(AGENT_ROOT / "lerobot" / ".venv" / "bin" / "python3"),
        "server_script": str(AGENT_ROOT / "lerobot" / "policy_server.py"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "lerobot"),
        ],
        "server_args": ["--checkpoint", "{checkpoint}", "--port", "{port}"],
        "env_vars": {},
        "arm_controller": "cartesian_pose",
        "policy_flag": "lerobot",
    },
    "diffusion_policy": {
        "server_python": str(AGENT_ROOT / "diffusion_policy" / ".venv" / "bin" / "python3"),
        "server_script": str(AGENT_ROOT / "diffusion_policy" / "policy_server.py"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "diffusion_policy"),
        ],
        "server_args": ["--checkpoint", "{checkpoint}", "--port", "{port}"],
        "env_vars": {},
        "arm_controller": "cartesian_pose",
        "policy_flag": "diffusion_policy",
    },
    "spatialvla": {
        "server_python": str(AGENT_ROOT / "SpatialVLA" / ".venv" / "bin" / "python3"),
        "server_script": str(AGENT_ROOT / "SpatialVLA" / "policy_server.py"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "SpatialVLA"),
        ],
        "server_args": ["--checkpoint", "{checkpoint}", "--port", "{port}"],
        "env_vars": {"HF_HOME": os.path.expanduser("~/.cache/huggingface")},
        "arm_controller": "cartesian_pose",
        "policy_flag": "spatialvla",
    },
}

# ---------------------------------------------------------------------------
# Known benchmarks — maps name → eval client + task_id + defaults
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS = {
    # LIBERO suites
    "libero_spatial": {"eval_client": "libero", "task_id": "libero_spatial", "max_steps": 220},
    "libero_object": {"eval_client": "libero", "task_id": "libero_object", "max_steps": 280},
    "libero_goal":   {"eval_client": "libero", "task_id": "libero_goal",   "max_steps": 300},
    "libero_10":     {"eval_client": "libero", "task_id": "libero_10",     "max_steps": 520},
    "libero_90":     {"eval_client": "libero", "task_id": "libero_90",     "max_steps": 400},
    # ManiSkill tasks (common ones)
    "maniskill:PickCube-v1":       {"eval_client": "maniskill", "task_id": "PickCube-v1",       "max_steps": 200},
    "maniskill:StackCube-v1":      {"eval_client": "maniskill", "task_id": "StackCube-v1",      "max_steps": 200},
    "maniskill:PegInsertionSide-v1": {"eval_client": "maniskill", "task_id": "PegInsertionSide-v1", "max_steps": 200},
    "maniskill:PickSingleYCB-v1":  {"eval_client": "maniskill", "task_id": "PickSingleYCB-v1",  "max_steps": 200},
    "maniskill:PushCube-v1":       {"eval_client": "maniskill", "task_id": "PushCube-v1",       "max_steps": 200},
    # RoboTwin tasks (common ones)
    "robotwin:beat_block_hammer":  {"eval_client": "robotwin", "task_id": "beat_block_hammer",   "max_steps": 300},
    "robotwin:handover_block":     {"eval_client": "robotwin", "task_id": "handover_block",      "max_steps": 300},
    "robotwin:stack_blocks_two":   {"eval_client": "robotwin", "task_id": "stack_blocks_two",    "max_steps": 300},
    "robotwin:place_bread_basket": {"eval_client": "robotwin", "task_id": "place_bread_basket",  "max_steps": 300},
    "robotwin:open_laptop":        {"eval_client": "robotwin", "task_id": "open_laptop",         "max_steps": 300},
    # RoboCasa tasks (common ones)
    "robocasa:PnPCounterToCab":    {"eval_client": "robocasa", "task_id": "PnPCounterToCab",     "max_steps": 300},
    "robocasa:PnPCabToCounter":    {"eval_client": "robocasa", "task_id": "PnPCabToCounter",     "max_steps": 300},
    "robocasa:PnPCounterToSink":   {"eval_client": "robocasa", "task_id": "PnPCounterToSink",    "max_steps": 300},
    "robocasa:PnPCounterToStove":  {"eval_client": "robocasa", "task_id": "PnPCounterToStove",   "max_steps": 300},
    "robocasa:OpenSingleDoor":     {"eval_client": "robocasa", "task_id": "OpenSingleDoor",      "max_steps": 200},
    "robocasa:CloseDrawer":        {"eval_client": "robocasa", "task_id": "CloseDrawer",         "max_steps": 200},
    "robocasa:TurnOnStove":        {"eval_client": "robocasa", "task_id": "TurnOnStove",         "max_steps": 200},
    "robocasa:TurnOffSinkFaucet":  {"eval_client": "robocasa", "task_id": "TurnOffSinkFaucet",   "max_steps": 200},
}


def run_preflight(policy_name, benchmark_name, checkpoint, allow_cross_domain=False):
    """Consult eval_registry to check readiness and auto-fix config.

    Returns:
        (ok, registry_cfg, warnings) where:
        - ok: True if evaluation should proceed
        - registry_cfg: EvalConfig from registry (or None)
        - warnings: list of warning strings
    """
    try:
        from policy_websocket.eval_registry import lookup, Readiness
    except ImportError:
        return True, None, ["eval_registry not available — skipping preflight"]

    # Map policy names to registry keys
    _POLICY_MAP = {
        "openvla": "openvla", "openvla-oft": "openvla",
        "pi0": "pi0", "pi0.5": "pi0.5", "smolvla": "smolvla",
        "spatialvla": "spatialvla",
        "lerobot": None,  # generic lerobot needs checkpoint-based detection
    }

    reg_policy = _POLICY_MAP.get(policy_name, policy_name)

    # For generic "lerobot", detect model from checkpoint name
    if reg_policy is None and checkpoint:
        ckpt_lower = checkpoint.lower()
        if "pi05" in ckpt_lower or "pi0.5" in ckpt_lower:
            reg_policy = "pi0.5"
        elif "pi0" in ckpt_lower:
            reg_policy = "pi0"
        elif "smolvla" in ckpt_lower:
            reg_policy = "smolvla"
        else:
            reg_policy = "lerobot"

    # Map benchmark to registry format
    reg_bench = benchmark_name
    if benchmark_name.startswith("libero_"):
        reg_bench = f"libero/{benchmark_name}"
    elif ":" in benchmark_name:
        # platform:task format (e.g. "maniskill:PickCube-v1") → try platform key
        reg_bench = benchmark_name.split(":")[0]

    cfg = lookup(reg_policy, reg_bench)
    if cfg is None:
        return True, None, [f"No registry entry for {reg_policy}×{reg_bench} — running without preflight"]

    warnings = []

    # Check readiness
    if cfg.readiness == Readiness.UNSUPPORTED:
        print(f"\n  PREFLIGHT BLOCKED: {cfg.summary()}")
        return False, cfg, ["UNSUPPORTED — known incompatibility, will not run"]

    if cfg.readiness == Readiness.NEEDS_FINETUNE:
        print(f"\n  PREFLIGHT WARNING: {cfg.summary()}")
        print(f"  This model needs finetuning for {reg_bench}. Expected: ~{cfg.expected_success_rate:.0%}")
        warnings.append(f"NEEDS_FINETUNE: expect ~{cfg.expected_success_rate:.0%}")
        if not allow_cross_domain:
            print(f"  Use --allow_cross_domain to run anyway.")
            return False, cfg, warnings
        print(f"  Running anyway (--allow_cross_domain)")

    if cfg.readiness == Readiness.CROSS_DOMAIN:
        print(f"\n  PREFLIGHT WARNING: {cfg.summary()}")
        print(f"  Cross-domain evaluation — expect ~{cfg.expected_success_rate:.0%}")
        warnings.append(f"CROSS_DOMAIN: expect ~{cfg.expected_success_rate:.0%}")
        if not allow_cross_domain:
            print(f"  Use --allow_cross_domain to run anyway.")
            return False, cfg, warnings
        print(f"  Running anyway (--allow_cross_domain)")

    if cfg.readiness == Readiness.READY:
        if cfg.expected_success_rate is not None:
            print(f"  Preflight: READY (expected {cfg.expected_success_rate:.0%} from {cfg.expected_source})")

    # Print known issues
    for issue in cfg.known_issues:
        warnings.append(issue)
        print(f"  Known issue: {issue}")

    # Note auto-resolution availability (actual resolution happens in main())
    if not checkpoint and cfg.checkpoint:
        warnings.append(f"Registry has checkpoint: {cfg.checkpoint}")

    # Check checkpoint mismatch (only when user explicitly provided one)
    if cfg.checkpoint and checkpoint and cfg.checkpoint != checkpoint:
        warnings.append(
            f"Checkpoint mismatch: you gave '{checkpoint}' but registry recommends '{cfg.checkpoint}'"
        )
        print(f"  WARNING: Registry recommends checkpoint '{cfg.checkpoint}'")
        print(f"           You provided '{checkpoint}'")

    return True, cfg, warnings


def resolve_benchmark(name):
    """Resolve benchmark config. Supports both known names and platform:task format.

    Examples:
        "libero_spatial"          → known config
        "maniskill:PickCube-v1"   → known config OR auto-generated
        "robotwin:open_laptop"    → known config OR auto-generated
    """
    # Check known configs first
    if name in BENCHMARK_CONFIGS:
        return BENCHMARK_CONFIGS[name]

    # Auto-detect from platform:task_id format
    if ":" in name:
        platform, task_id = name.split(":", 1)
        if platform in EVAL_CLIENTS:
            return {
                "eval_client": platform,
                "task_id": task_id,
                "max_steps": 300,  # default
            }

    # Backward compat: bare libero_ names
    if name.startswith("libero_"):
        return {"eval_client": "libero", "task_id": name, "max_steps": 300}

    return None


def resolve_policy(name):
    """Resolve policy config, following aliases."""
    cfg = POLICY_CONFIGS.get(name)
    if cfg and "alias" in cfg:
        return POLICY_CONFIGS[cfg["alias"]]
    return cfg


def start_server(policy_cfg, checkpoint, port, gpu_id, node=None):
    """Start the policy server, return subprocess."""
    python = policy_cfg["server_python"]
    script = policy_cfg["server_script"]
    pythonpath = ":".join(policy_cfg["server_pythonpath"])

    args = []
    for a in policy_cfg["server_args"]:
        args.append(a.replace("{checkpoint}", checkpoint).replace("{port}", str(port)))

    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for k, v in policy_cfg.get("env_vars", {}).items():
        env[k] = v

    cmd = [python, script] + args

    if node:
        env_exports = " ".join(f"{k}={v}" for k, v in [
            ("PYTHONPATH", pythonpath),
            ("CUDA_VISIBLE_DEVICES", str(gpu_id)),
            ("MUJOCO_GL", "egl"),
        ] + [(k, v) for k, v in policy_cfg.get("env_vars", {}).items()])

        srun_cmd = [
            "srun", "--overlap", f"--nodelist={node}", "--gres=gpu:1",
            "--time=01:00:00", "--job-name=policy-server",
            "bash", "-c", f"{env_exports} {python} {script} {' '.join(args)}"
        ]
        print(f"  Starting server on {node} (GPU {gpu_id}, port {port})...")
        proc = subprocess.Popen(
            srun_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
    else:
        print(f"  Starting server locally (GPU {gpu_id}, port {port})...")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )

    return proc


def wait_for_server(port, timeout=600, interval=10, proc=None):
    """Wait for server healthcheck to respond."""
    import urllib.request
    url = f"http://localhost:{port}/healthz"
    start = time.time()
    while time.time() - start < timeout:
        if proc and proc.poll() is not None:
            print(f"  Server process died (exit code {proc.returncode})")
            return False
        try:
            urllib.request.urlopen(url, timeout=5)
            elapsed = int(time.time() - start)
            print(f"  Server ready after {elapsed}s")
            return True
        except Exception:
            time.sleep(interval)
    print(f"  Server not ready after {timeout}s")
    return False


def run_eval(benchmark_cfg, policy_cfg, port, num_trials=5, log_dir=None,
             server_host="localhost", policy_name="policy"):
    """Run evaluation using the appropriate eval client.

    This is the universal eval runner — works for LIBERO, ManiSkill, RoboTwin,
    or any platform with a WebSocket eval script.
    """
    if log_dir is None:
        log_dir = str(AGENT_ROOT / "logs" / "eval_results")

    client_name = benchmark_cfg["eval_client"]
    client = EVAL_CLIENTS[client_name]

    eval_python = client["eval_python"]
    eval_script = client["eval_script"]
    pythonpath = ":".join(client["eval_pythonpath"])

    # Build argument list by substituting template variables
    server_addr = f"{server_host}:{port}"
    arm_controller = policy_cfg.get("arm_controller", "cartesian_pose") if policy_cfg else "cartesian_pose"
    policy_flag = policy_cfg.get("policy_flag", policy_name) if policy_cfg else policy_name

    # Map arm_controller to RoboTwin action_type
    _AC_TO_ACTION_TYPE = {"cartesian_pose": "ee", "joint_vel": "qpos", "joint_pos": "qpos"}
    action_type = _AC_TO_ACTION_TYPE.get(arm_controller, "ee")

    subs = {
        "{server_addr}": server_addr,
        "{policy_flag}": policy_flag,
        "{task_id}": benchmark_cfg["task_id"],
        "{num_trials}": str(num_trials),
        "{arm_controller}": arm_controller,
        "{action_type}": action_type,
        "{log_dir}": log_dir,
    }

    cmd_args = []
    for a in client["eval_args"]:
        for k, v in subs.items():
            a = a.replace(k, v)
        cmd_args.append(a)

    cmd = [eval_python, "-u", eval_script] + cmd_args

    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    for k, v in client.get("env_vars", {}).items():
        env[k] = v

    task_id = benchmark_cfg["task_id"]
    print(f"  Running eval: {client_name}:{task_id}, {num_trials} trials...")
    print(f"  Policy: {policy_flag}, controller: {arm_controller}")

    # Write eval output to a log file
    eval_log = Path(log_dir) / f"eval_{client_name}_{task_id}_latest.log"
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    print(f"  Eval log: {eval_log}")

    with open(eval_log, "w") as logf:
        result = subprocess.run(
            cmd, stdout=logf, stderr=subprocess.STDOUT,
            text=True, env=env, timeout=7200,
        )

    output = eval_log.read_text()

    # Parse success rate from output (universal pattern)
    success_rate = None
    for line in output.split("\n"):
        line_lower = line.lower()
        if any(kw in line_lower for kw in ("overall success", "average success", "success rate")):
            m = re.search(r"(\d+\.?\d*)%", line)
            if m:
                success_rate = float(m.group(1))

    return {
        "exit_code": result.returncode,
        "success_rate": success_rate,
        "output": output[-3000:],
    }


def submit_as_slurm_job(policy_cfg, args, benchmark_cfg):
    """Submit the entire benchmark as a SLURM job (server + eval in one job)."""
    log_dir = args.log_dir or str(AGENT_ROOT / "logs" / "eval_results")
    job_log = str(AGENT_ROOT / "logs" / f"benchmark-{args.policy}-{args.benchmark}-%j.log")

    python_server = policy_cfg["server_python"]
    script_server = policy_cfg["server_script"]
    pp_server = ":".join(policy_cfg["server_pythonpath"])

    server_args = []
    for a in policy_cfg["server_args"]:
        server_args.append(a.replace("{checkpoint}", args.checkpoint).replace("{port}", str(args.port)))

    extra_env_exports = []
    for k, v in policy_cfg.get("env_vars", {}).items():
        extra_env_exports.append(f"export {k}={v}")
    extra_env_block = "\n".join(extra_env_exports)

    # Resolve eval client
    client_name = benchmark_cfg["eval_client"]
    client = EVAL_CLIENTS[client_name]
    eval_python = client["eval_python"]
    eval_script = client["eval_script"]
    pp_eval = ":".join(client["eval_pythonpath"])

    # Build eval arguments
    arm_controller = policy_cfg.get("arm_controller", "cartesian_pose")
    policy_flag = policy_cfg.get("policy_flag", args.policy)
    task_id = benchmark_cfg["task_id"]

    _AC_TO_ACTION_TYPE = {"cartesian_pose": "ee", "joint_vel": "qpos", "joint_pos": "qpos"}
    action_type = _AC_TO_ACTION_TYPE.get(arm_controller, "ee")

    subs = {
        "{server_addr}": f"localhost:{args.port}",
        "{policy_flag}": policy_flag,
        "{task_id}": task_id,
        "{num_trials}": str(args.num_trials),
        "{arm_controller}": arm_controller,
        "{action_type}": action_type,
        "{log_dir}": log_dir,
    }
    eval_cmd_args = []
    for a in client["eval_args"]:
        for k, v in subs.items():
            a = a.replace(k, v)
        eval_cmd_args.append(a)
    eval_args_str = " \\\n  ".join(eval_cmd_args)

    # Eval-specific env vars
    eval_env_exports = []
    for k, v in client.get("env_vars", {}).items():
        eval_env_exports.append(f"export {k}={v}")
    eval_env_block = "\n".join(eval_env_exports)

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=bench-{args.policy[:8]}
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output={job_log}
{"#SBATCH --nodelist=" + args.node if args.node else ""}
#SBATCH --exclude={EGL_EXCLUDE_NODES}

{extra_env_block}

echo "=== Benchmark: {args.policy} on {args.benchmark} ==="
echo "Eval client: {client_name}, Task: {task_id}"
echo "Node: $(hostname), GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

# Step 1: Start server (CUDA_VISIBLE_DEVICES as prefix, not global)
CUDA_VISIBLE_DEVICES={args.gpu_id} PYTHONPATH={pp_server} {python_server} {script_server} {' '.join(server_args)} &
SERVER_PID=$!

# Step 2: Wait for server
for i in $(seq 1 60); do
  curl -s http://localhost:{args.port}/healthz > /dev/null 2>&1 && {{ echo "Server ready after ${{i}}0s"; break; }}
  kill -0 $SERVER_PID 2>/dev/null || {{ echo "Server died"; exit 1; }}
  sleep 10
done

# Step 3: Run eval
echo "=== Running eval ({client_name}) ==="
{eval_env_block}
PYTHONPATH={pp_eval} {eval_python} -u {eval_script} \\
  {eval_args_str}
EVAL_EXIT=$?

echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
exit $EVAL_EXIT
"""

    script_path = AGENT_ROOT / "logs" / f"_bench_{args.policy}_{args.benchmark.replace(':', '_')}.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(sbatch_script)
    script_path.chmod(0o755)

    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True, text=True, timeout=30,
    )
    return result.stdout.strip(), str(script_path), job_log


def list_benchmarks():
    """Print all known benchmarks grouped by platform."""
    print("Available benchmarks:")
    by_platform = {}
    for name, cfg in BENCHMARK_CONFIGS.items():
        platform = cfg["eval_client"]
        by_platform.setdefault(platform, []).append(name)

    for platform, names in sorted(by_platform.items()):
        print(f"\n  {platform.upper()}:")
        for n in sorted(names):
            print(f"    {n}")

    print(f"\n  Custom format: <platform>:<task_id>")
    print(f"  Platforms: {', '.join(sorted(EVAL_CLIENTS.keys()))}")
    print(f"\n  Examples:")
    print(f"    maniskill:PickCube-v1")
    print(f"    robotwin:beat_block_hammer")
    print(f"    libero_spatial")


def main():
    parser = argparse.ArgumentParser(description="Run benchmark evaluation end-to-end (multi-platform)")
    parser.add_argument("--policy", required=True, help="Policy name (e.g. openvla, diffusion_policy)")
    parser.add_argument("--checkpoint", default=None,
                        help="Model checkpoint path or HF model ID (auto-resolved from registry if omitted)")
    parser.add_argument("--benchmark", required=True,
                        help="Benchmark: libero_spatial, maniskill:PickCube-v1, robotwin:beat_block_hammer")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of trials per task")
    parser.add_argument("--port", type=int, default=18800, help="Server port")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU device ID")
    parser.add_argument("--node", default=None, help="Compute node (if remote)")
    parser.add_argument("--server_addr", default=None,
                        help="Connect to existing server (host:port), skip server startup")
    parser.add_argument("--log_dir", default=None, help="Log directory")
    parser.add_argument("--submit", action="store_true",
                        help="Submit as SLURM job instead of running locally")
    parser.add_argument("--list_benchmarks", action="store_true",
                        help="List all available benchmarks and exit")
    parser.add_argument("--allow_cross_domain", action="store_true",
                        help="Allow running NEEDS_FINETUNE / CROSS_DOMAIN combos (expect ~0%%)")
    parser.add_argument("--skip_preflight", action="store_true",
                        help="Skip eval_registry preflight check")
    parser.add_argument("--unnorm_key", default=None,
                        help="Unnorm key for OpenVLA (auto-detected from registry if not set)")
    args = parser.parse_args()

    if args.list_benchmarks:
        list_benchmarks()
        return

    # Resolve policy config
    policy_cfg = resolve_policy(args.policy)
    if policy_cfg is None:
        for yaml_path in AGENT_ROOT.glob("*/policy_server.yaml"):
            import yaml
            meta = yaml.safe_load(yaml_path.read_text())
            if meta.get("name") == args.policy:
                repo = yaml_path.parent
                venv_python = str(repo / ".venv" / "bin" / "python3")
                ps = meta.get("policy_server", {})
                # Build server args from yaml arguments list
                srv_args = []
                for arg_def in ps.get("arguments", []):
                    flag = arg_def.get("flag", f"--{arg_def['name']}")
                    srv_args.extend([flag, "{" + arg_def["name"] + "}"])
                if not srv_args:
                    srv_args = ["--checkpoint", "{checkpoint}", "--port", "{port}"]
                # Read arm_controller from yaml resources
                resources = ps.get("resources", {})
                arm_controller = resources.get("arm_controller", "cartesian_pose")
                # Read entry_point
                entry_point = ps.get("entry_point", "policy_server.py")
                # Read pythonpath
                setup = ps.get("setup", {})
                pp_raw = setup.get("pythonpath", [])
                pythonpath = [str(AGENT_ROOT / p) if not p.startswith("/") else p for p in pp_raw]
                if not pythonpath:
                    pythonpath = [str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"), str(repo)]
                # Read env vars
                env_vars = setup.get("env_vars", {})
                policy_cfg = {
                    "server_python": venv_python,
                    "server_script": str(repo / entry_point),
                    "server_pythonpath": pythonpath,
                    "server_args": srv_args,
                    "env_vars": env_vars,
                    "arm_controller": arm_controller,
                    "policy_flag": args.policy,
                }
                break
        if policy_cfg is None:
            print(f"ERROR: Unknown policy '{args.policy}'. Known: {list(POLICY_CONFIGS.keys())}")
            print("Or provide a repo with policy_server.yaml")
            sys.exit(1)

    # Resolve benchmark config
    benchmark_cfg = resolve_benchmark(args.benchmark)
    if benchmark_cfg is None:
        print(f"ERROR: Unknown benchmark '{args.benchmark}'.")
        list_benchmarks()
        sys.exit(1)

    client_name = benchmark_cfg["eval_client"]
    if client_name not in EVAL_CLIENTS:
        print(f"ERROR: No eval client '{client_name}'. Available: {list(EVAL_CLIENTS.keys())}")
        sys.exit(1)

    # --- Preflight: consult eval_registry ---
    if not args.skip_preflight:
        ok, reg_cfg, warnings = run_preflight(
            args.policy, args.benchmark, args.checkpoint,
            allow_cross_domain=args.allow_cross_domain,
        )
        for w in warnings:
            print(f"  [preflight] {w}")
        if not ok:
            print(f"\nPreflight blocked this evaluation. Use --allow_cross_domain or --skip_preflight to override.")
            sys.exit(2)

        # Auto-resolve checkpoint from registry when not provided by user
        if not args.checkpoint and reg_cfg and reg_cfg.checkpoint:
            args.checkpoint = reg_cfg.checkpoint
            print(f"  Auto-resolved checkpoint: {args.checkpoint}")

        # Auto-apply registry server_args (e.g. --unnorm_key for OpenVLA)
        if reg_cfg and reg_cfg.server_args:
            for flag, val in reg_cfg.server_args.items():
                # Don't override checkpoint — handled separately above
                if flag == "--pretrained_checkpoint" or flag == "--checkpoint":
                    continue
                # Check if this arg is already in the policy config
                flat_args = " ".join(policy_cfg.get("server_args", []))
                if flag not in flat_args:
                    policy_cfg.setdefault("server_args", []).extend([flag, val])
                    print(f"  Auto-added from registry: {flag} {val}")

        # Auto-apply arm_controller from registry (overrides policy default)
        if reg_cfg and reg_cfg.arm_controller:
            old_ac = policy_cfg.get("arm_controller", "")
            if old_ac != reg_cfg.arm_controller:
                policy_cfg["arm_controller"] = reg_cfg.arm_controller
                print(f"  Auto-override arm_controller: {old_ac} → {reg_cfg.arm_controller}")

        # Auto-apply --unnorm_key for OpenVLA if not explicitly set
        if args.unnorm_key:
            flat_args = " ".join(policy_cfg.get("server_args", []))
            if "--unnorm_key" not in flat_args:
                policy_cfg["server_args"].extend(["--unnorm_key", args.unnorm_key])
                print(f"  Added --unnorm_key {args.unnorm_key}")

    # Final checkpoint check — fallback to a default per policy for cross-domain
    if not args.checkpoint:
        _DEFAULT_CHECKPOINTS = {
            "openvla": "moojink/openvla-7b-oft-finetuned-libero-spatial",
            "pi0": "lerobot/pi0_libero_finetuned",
            "pi0.5": "lerobot/pi05_libero_finetuned",
            "smolvla": "lerobot/smolvla_base",
            "spatialvla": "IPEC-COMMUNITY/spatialvla-4b-224-pt",
        }
        resolved = resolve_policy(args.policy)
        policy_key = args.policy
        fallback = _DEFAULT_CHECKPOINTS.get(policy_key)
        if fallback:
            args.checkpoint = fallback
            print(f"  Fallback checkpoint (no registry match): {args.checkpoint}")
        else:
            print("ERROR: No --checkpoint provided and registry has no checkpoint for this combo.")
            print("  Provide --checkpoint explicitly or check eval_registry entries.")
            sys.exit(1)

    print(f"\n=== Benchmark Run ===")
    print(f"Policy:     {args.policy}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Benchmark:  {args.benchmark}")
    print(f"Platform:   {client_name}")
    print(f"Task ID:    {benchmark_cfg['task_id']}")
    print(f"Trials:     {args.num_trials}")

    # Mode 1: Submit as SLURM job
    has_slurm = subprocess.run(["which", "sbatch"], capture_output=True).returncode == 0
    if args.submit or (args.node and not args.server_addr):
        if not has_slurm:
            print("SLURM not available — falling back to local execution mode")
            args.submit = False
            args.node = None
        else:
            print(f"Mode: SLURM job submission")
            print()
            sbatch_out, script_path, job_log = submit_as_slurm_job(policy_cfg, args, benchmark_cfg)
            print(f"Submitted: {sbatch_out}")
            print(f"Script: {script_path}")
            print(f"Log: {job_log.replace('%j', '<JOBID>')}")
            print(f"\nMonitor with: tail -f {job_log.replace('%j', '<JOBID>')}")
            print(f"Check status: squeue -u $USER")
            return

    # Mode 2: Connect to existing server
    if args.server_addr:
        host, port = args.server_addr.split(":")
        port = int(port)
        print(f"Mode: Connect to existing server at {host}:{port}")
        print()
        eval_result = run_eval(
            benchmark_cfg=benchmark_cfg,
            policy_cfg=policy_cfg,
            port=port,
            server_host=host,
            num_trials=args.num_trials,
            log_dir=args.log_dir,
            policy_name=args.policy,
        )
        print(f"\n=== Results ===")
        print(f"Exit code: {eval_result['exit_code']}")
        if eval_result["success_rate"] is not None:
            print(f"Success rate: {eval_result['success_rate']}%")
        print(f"\nLast output:\n{eval_result['output'][-500:]}")
        summary = {
            "policy": args.policy, "benchmark": args.benchmark,
            "platform": client_name, "task_id": benchmark_cfg["task_id"],
            "success_rate": eval_result["success_rate"],
        }
        print(f"\n{json.dumps(summary, indent=2)}")
        sys.exit(0 if eval_result["exit_code"] == 0 else 1)

    # Mode 3: Local (start server + run eval in same process)
    print(f"Mode: Local execution")
    print()
    server_proc = start_server(policy_cfg, args.checkpoint, args.port, args.gpu_id)

    try:
        if not wait_for_server(args.port, timeout=600, proc=server_proc):
            print("ERROR: Server failed to start")
            server_proc.kill()
            sys.exit(1)

        eval_result = run_eval(
            benchmark_cfg=benchmark_cfg,
            policy_cfg=policy_cfg,
            port=args.port,
            num_trials=args.num_trials,
            log_dir=args.log_dir,
            policy_name=args.policy,
        )

        print(f"\n=== Results ===")
        print(f"Exit code: {eval_result['exit_code']}")
        if eval_result["success_rate"] is not None:
            print(f"Success rate: {eval_result['success_rate']}%")
        print(f"\nLast output:\n{eval_result['output'][-500:]}")

        summary = {
            "policy": args.policy,
            "checkpoint": args.checkpoint,
            "benchmark": args.benchmark,
            "platform": client_name,
            "task_id": benchmark_cfg["task_id"],
            "num_trials": args.num_trials,
            "exit_code": eval_result["exit_code"],
            "success_rate": eval_result["success_rate"],
        }
        print(f"\n{json.dumps(summary, indent=2)}")

    finally:
        print("\nStopping server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("Done.")


if __name__ == "__main__":
    main()
