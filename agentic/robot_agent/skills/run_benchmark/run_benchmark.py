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
import random
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

# MuJoCo EGL rendering only works on a subset of nodes (driver/firmware dependent).
# LIBERO and RoboCasa (robosuite) need EGL; ManiSkill (SAPIEN/Vulkan) is unaffected.
EGL_GOOD_NODES = os.environ.get("EGL_GOOD_NODES", "cn24,cn25,cn26,cn27")
EGL_EXCLUDE_NODES = os.environ.get("EGL_EXCLUDE_NODES", "")
# ManiSkill sapien/Vulkan hangs silently on these nodes at "Episode 1/10" —
# verified 2026-04-20 across octo + openvla; sister jobs on cn13/cn21/cn35 fine.
# RoboTwin also uses sapien + curobo CUDA graphs which hit "CUDA illegal instruction"
# on the same bad nodes (cn19 confirmed 2026-04-24 in job 116889).
MANISKILL_EXCLUDE_NODES = os.environ.get("MANISKILL_EXCLUDE_NODES", "cn16,cn17,cn19")
ROBOTWIN_EXCLUDE_NODES = os.environ.get("ROBOTWIN_EXCLUDE_NODES", "cn16,cn17,cn19")

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
        # curobo prebuilt .so in RoboTwin/.venv only has sm_80; H100 is sm_90 and
        # aborts with "CUDA error: illegal instruction" inside line_search_cu. We
        # rebuilt the 5 CUDA extensions via torch JIT with spack cuda@12.6 +
        # TORCH_CUDA_ARCH_LIST="8.0 9.0" into curobo_ext_cache/. At runtime, the
        # stripped site-packages .so re-triggers JIT load and finds the cached
        # compile — no recompile, just loads the sm_90 .so. CUDA_HOME/PATH are
        # set so a cache miss (cold node) still finds a working nvcc.
        "env_vars": {
            "CUDA_HOME": "/mnt/vast/spack/v0.23/opt/spack/linux-rocky9-sapphirerapids/gcc-13.3.0/cuda-12.6.2-iipq3kx6jniy56k3iqzxqlccmnl4tgt7",
            "PATH": "$CUDA_HOME/bin:$PATH",
            "LD_LIBRARY_PATH": "$CUDA_HOME/lib64:$LD_LIBRARY_PATH",
            "TORCH_EXTENSIONS_DIR": str(AGENT_ROOT / "RoboTwin" / ".venv" / "curobo_ext_cache"),
            "TORCH_CUDA_ARCH_LIST": '"8.0 9.0"',
        },
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
    "pi0fast": {"alias": "lerobot"},
    "pi0_fast": {"alias": "lerobot"},
    "smolvla": {"alias": "lerobot"},
    # RDT-1B (Robotics Diffusion Transformer) — separate repo + venv from lerobot.
    # Own server_script handles ManiSkill single-arm and RoboTwin bimanual modes.
    "rdt": {
        "server_python": str(AGENT_ROOT / "RDT" / ".venv" / "bin" / "python3"),
        "server_script": str(AGENT_ROOT / "RDT" / "policy_server.py"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "RDT"),
        ],
        "server_args": ["--checkpoint", "{checkpoint}", "--port", "{port}"],
        "env_vars": {},
        "arm_controller": "joint_pos",
        "policy_flag": "rdt",
    },
    # ACT (DETR-CVAE) — RoboTwin upstream policy, own venv (torch 2.4.1+cu121 for sm_90).
    "act": {
        "server_python": str(AGENT_ROOT / "RoboTwin" / "policy" / "ACT" / ".venv" / "bin" / "python3"),
        "server_script": str(AGENT_ROOT / "RoboTwin" / "policy" / "ACT" / "policy_server.py"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "RoboTwin" / "policy" / "ACT"),
        ],
        "server_args": ["--checkpoint", "{checkpoint}", "--port", "{port}"],
        "env_vars": {},
        "arm_controller": "joint_pos",
        "policy_flag": "act",
    },
    # DP (Diffusion Policy) — RoboTwin upstream policy, own venv (hydra+dill+diffusers).
    "dp": {
        "server_python": str(AGENT_ROOT / "RoboTwin" / "policy" / "DP" / ".venv" / "bin" / "python3"),
        "server_script": str(AGENT_ROOT / "RoboTwin" / "policy" / "DP" / "policy_server.py"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "RoboTwin" / "policy" / "DP"),
        ],
        "server_args": ["--checkpoint", "{checkpoint}", "--port", "{port}"],
        "env_vars": {},
        "arm_controller": "joint_pos",
        "policy_flag": "dp",
    },
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
    # OpenPI (JAX/orbax) — handles RLinf, kimtaey, SakikoTogawa-style finetuned
    # pi0/pi0.5/pi0_fast checkpoints that use the openpi training pipeline.
    "openpi": {
        "server_python": str(AGENT_ROOT / "openpi" / ".venv" / "bin" / "python3"),
        "server_script": str(AGENT_ROOT / "openpi" / "scripts" / "policy_server.py"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "openpi" / "src"),
        ],
        # --config is filled in by route_openpi_config() from the benchmark
        "server_args": ["--config", "{openpi_config}", "--checkpoint", "{checkpoint}", "--port", "{port}"],
        # System python3.11 at /usr/include/python3.11 only ships pyconfig-64.h;
        # triton JIT needs Python.h when inductor compiles launcher kernels for
        # pi0.5 PyTorch. UV's full python distribution has the headers — point
        # CPATH there so inductor's gcc invocation resolves `#include <Python.h>`.
        "env_vars": {
            "CPATH": "/mnt/vast/home/yd66byne/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/include/python3.11:$CPATH",
            "C_INCLUDE_PATH": "/mnt/vast/home/yd66byne/.local/share/uv/python/cpython-3.11.15-linux-x86_64-gnu/include/python3.11:$C_INCLUDE_PATH",
        },
        "arm_controller": "joint_vel",
        "policy_flag": "openpi",
    },
}


def resolve_lerobot_nested_ckpt(ckpt_path: str) -> str:
    """LeRobot training writes config.json to checkpoints/<step>/pretrained_model/.
    The policy_server passes the directory straight to HuggingFace's from_pretrained,
    which needs config.json at the top level. If top-level is missing but a nested
    pretrained_model/ exists with one, resolve to the latest-step nested path.
    Why: 2026-04-20 pi0-libero_spatial-20260414_133536/ job 115755 died with
    draccus ParsingError because top-level config.json was absent (nested only).
    """
    p = Path(ckpt_path)
    if not p.is_dir() or (p / "config.json").is_file():
        return ckpt_path
    ckpts_dir = p / "checkpoints"
    if not ckpts_dir.is_dir():
        return ckpt_path
    step_dirs = sorted(
        [d for d in ckpts_dir.iterdir() if d.is_dir() and (d / "pretrained_model" / "config.json").is_file()],
        key=lambda d: d.name,
    )
    if not step_dirs:
        return ckpt_path
    resolved = step_dirs[-1] / "pretrained_model"
    print(f"  [resolve] nested lerobot ckpt → {resolved}")
    return str(resolved)


def is_openpi_checkpoint(ckpt_path: str) -> bool:
    """Detect OpenPI/orbax-trained checkpoint. Several layouts exist in the wild:
    - Native orbax (kimtaey, RLinf RL): `_CHECKPOINT_METADATA` + `params/`
    - HuggingFace-converted openpi (RLinf SFT): flat `model.safetensors` next
      to `physical-intelligence/<task>/norm_stats.json`
    - Subdir layouts: orbax under any child dir
    """
    p = Path(ckpt_path)
    if not p.is_dir():
        return False
    if (p / "_CHECKPOINT_METADATA").is_file():
        return True
    if (p / "params").is_dir() and (p / "assets").is_dir():
        return True
    # HuggingFace mirror of openpi training: "physical-intelligence/" directory
    # holds the per-task norm_stats.json that openpi expects.
    if (p / "physical-intelligence").is_dir():
        return True
    # Some uploads put it under a subdir
    for sub in p.iterdir():
        if sub.is_dir() and ((sub / "_CHECKPOINT_METADATA").is_file() or (sub / "params").is_dir()):
            return True
    return False


def _detect_pi_family(ckpt_path: str | None) -> str:
    """Detect pi-family architecture from checkpoint contents.
    Returns 'pi05', 'pi0_fast', or 'pi0'. The stock openpi server's load path
    branches on this — Pi0FASTConfig lacks a `.pi05` attribute, so routing a
    pi0.5 safetensors ckpt to a pi0_fast config crashes.
    """
    if not ckpt_path:
        return "pi0_fast"
    p = Path(ckpt_path)
    name = p.name.lower()
    # Strongest signal: explicit family token in path
    if "pi05" in name or "pi0.5" in name or "pi0_5" in name:
        return "pi05"
    if "pi0_fast" in name or "pi0fast" in name or "pi-fast" in name:
        return "pi0_fast"
    # Inspect metadata.pt / config.json hints
    for cfg_name in ("config.json", "metadata.json"):
        cfg_path = p / cfg_name
        if cfg_path.is_file():
            try:
                txt = cfg_path.read_text(errors="ignore").lower()
                if "pi05" in txt or "pi0.5" in txt:
                    return "pi05"
                if "pi0_fast" in txt or "pi0fast" in txt:
                    return "pi0_fast"
            except Exception:
                pass
    return "pi0_fast"  # safe default — most community ckpts are pi0_fast


def route_openpi_config(benchmark_name: str, ckpt_path: str | None = None) -> str:
    """Pick the closest stock openpi config for a given benchmark+family.
    The training teams use custom configs that aren't in stock openpi, but the
    model arch is fully specified by the weights — a matching stock config
    (action_dim, horizon, robot type, family) is enough for inference.

    Family routing avoids the `Pi0FASTConfig has no attribute 'pi05'` crash:
      pi05      → pi05_libero / pi05_droid
      pi0_fast  → pi0_fast_libero / pi0_fast_droid
      pi0       → pi0_libero / pi0_droid
    Benchmark routing (per family):
      libero/maniskill/robocasa/calvin → *_libero (Franka, 7 DoF)
      droid                             → *_droid  (8 DoF)
    """
    family = _detect_pi_family(ckpt_path)
    bm = benchmark_name.split(":")[0].lower()
    bench_slot = "droid" if bm == "droid" else "libero"
    return f"{family}_{bench_slot}"

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


def _protocol_server_extras(user_policy_name: str, benchmark_name: str) -> list[str]:
    """Read eval_protocol.json and return extra server-side CLI args.

    Catches the class of bug where protocol knows image_flip_180deg=false and
    center_crop=false for openvla/ManiSkill but generated server command still
    runs with default flip+center_crop (PullCube 80% → 0%).
    """
    agent_root = os.environ.get("AGENTROBOT_ROOT", str(AGENT_ROOT))
    bench_short = benchmark_name.split(":")[0] if ":" in benchmark_name else benchmark_name
    if bench_short.startswith("libero_"):
        bench_short = "libero"
    proto_path = Path(agent_root) / "agentic/robot_agent/eval_protocols" / f"{user_policy_name}_{bench_short}.json"
    if not proto_path.is_file():
        return []
    try:
        fields = json.loads(proto_path.read_text()).get("fields", {})
    except Exception:
        return []

    def _v(key):
        f = fields.get(key)
        return f.get("value") if isinstance(f, dict) else None

    extras: list[str] = []
    if user_policy_name in ("openvla", "openvla-oft"):
        if _v("image_flip_180deg") is False:
            extras += ["--no_flip_image"]
        if _v("center_crop") is False:
            extras += ["--no_center_crop"]
        if _v("invert_gripper") is False:
            extras += ["--no_invert_gripper"]
    return extras


def _protocol_cli_extras(client_name: str, user_policy_name: str, benchmark_name: str) -> list[str]:
    """Read eval_protocol.json and return extra CLI args for the given client.

    Catches the class of bug where the protocol knows max_episode_steps=300 and
    HumanCameraWrapper is required but the generated eval command uses env defaults
    (PullCube-v1 env spec = 40 steps) → 0% success despite correct policy.
    """
    agent_root = os.environ.get("AGENTROBOT_ROOT", str(AGENT_ROOT))
    bench_short = benchmark_name.split(":")[0] if ":" in benchmark_name else benchmark_name
    if bench_short.startswith("libero_"):
        bench_short = "libero"
    proto_path = Path(agent_root) / "agentic/robot_agent/eval_protocols" / f"{user_policy_name}_{bench_short}.json"
    if not proto_path.is_file():
        return []
    try:
        fields = json.loads(proto_path.read_text()).get("fields", {})
    except Exception:
        return []

    def _v(key):
        f = fields.get(key)
        return f.get("value") if isinstance(f, dict) else None

    extras: list[str] = []
    if client_name == "maniskill":
        mes = _v("max_episode_steps")
        if isinstance(mes, int):
            extras += ["--max_episode_steps", str(mes)]
        res = _v("image_resolution")
        if isinstance(res, int):
            extras += ["--img_res", str(res)]
        cam = _v("camera")
        if isinstance(cam, str) and "HumanCamera" in cam:
            extras += ["--use_human_camera"]
        cm = _v("control_mode")
        if isinstance(cm, str) and cm:
            extras += ["--control_mode", cm]
    elif client_name == "robocasa":
        res = _v("image_resolution")
        if isinstance(res, int):
            extras += ["--img_res", str(res)]
    elif client_name == "robotwin":
        # Auto-injected --seed_base when train/eval seed ranges were disjoint.
        # Skill emits recommended_client_flags after overlap autofix; pass them
        # through so RoboTwin's run_eval_ws.py starts episodes inside the
        # ckpt's training seed range instead of at 100000 (OOD by default).
        rec = _v("recommended_client_flags")
        if isinstance(rec, list) and all(isinstance(x, str) for x in rec):
            extras += rec
    return extras


# Canonical state_dim emitted by each benchmark's obs_to_policy_format pipeline.
# Pre-submission gate compares checkpoint-claimed state_dim vs this table.
# Catches pi0fast-bimanual(16D) × RoboCasa-singlearm(8D) BEFORE SLURM submission
# instead of IndexError at inference after 2+ min of checkpoint load.
_SIM_STATE_DIM = {
    "maniskill": 8,    # Panda: eef_pos(3) + eef_quat(4) or gripper(1) or qpos(7+1)
    "robocasa":  8,    # Single-arm Franka
    "robotwin": 14,    # Bimanual (2x7)
    "libero":    8,    # Single-arm Franka, LIBERO spec
}


def run_protocol_gate(policy_name: str, benchmark_name: str, checkpoint: str | None):
    """Validate eval_protocol.json exists and is fully cited for this combo.

    Returns (ok, message). Without a passing gate, run_benchmark refuses to
    submit — every past misalignment (center_crop, human_render_camera, no-flip,
    no-invert-gripper) would have been caught here if the gate existed.
    """
    agent_root = os.environ.get("AGENTROBOT_ROOT", str(AGENT_ROOT))
    # normalize benchmark (strip :task_id)
    bench_short = benchmark_name.split(":")[0] if ":" in benchmark_name else benchmark_name
    if bench_short.startswith("libero_"):
        bench_short = "libero"
    proto_path = Path(agent_root) / "agentic/robot_agent/eval_protocols" / f"{policy_name}_{bench_short}.json"
    skill = Path(agent_root) / "agentic/robot_agent/skills/extract_eval_protocol/extract_eval_protocol.py"
    if not skill.is_file():
        return True, f"extract_eval_protocol skill missing ({skill}) — gate disabled"
    result = subprocess.run(
        [sys.executable, str(skill),
         "--policy", policy_name, "--benchmark", bench_short,
         "--checkpoint", checkpoint or "",
         "--out", str(proto_path),
         "--validate"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return False, (result.stderr or result.stdout).strip()

    # Shape-mismatch sub-gate: checkpoint state_dim vs simulator canonical dim.
    sim_dim = _SIM_STATE_DIM.get(bench_short)
    if sim_dim is not None and proto_path.is_file():
        try:
            fields = json.loads(proto_path.read_text()).get("fields", {})
            ckpt_sd_field = fields.get("state_dim")
            ckpt_sd = ckpt_sd_field.get("value") if isinstance(ckpt_sd_field, dict) else None
            if isinstance(ckpt_sd, int) and ckpt_sd != sim_dim:
                return False, (
                    f"state_dim mismatch: checkpoint expects {ckpt_sd}D, "
                    f"{bench_short} simulator emits {sim_dim}D. "
                    f"Submission would crash at inference "
                    f"(shape ({sim_dim},) vs ({ckpt_sd},)). "
                    f"Pick a checkpoint trained on a matching embodiment, "
                    f"or choose a benchmark whose state_dim matches {ckpt_sd}."
                )
        except Exception:
            pass

    return True, f"✓ eval_protocol.json validated: {proto_path}"


# Safetensors tensor-key markers per policy class. `require` = substrings that
# MUST appear in at least one state_dict key; `forbid` = substrings that MUST
# NOT appear (with a human-readable hint explaining what the forbidden tensors
# imply about the true underlying policy). Calibrated 2026-04-24 against
# lerobot/pi0_libero_finetuned, lerobot/pi05_libero_finetuned,
# lerobot/pi0fast-libero, jadechoghari/pi0fast-libero.
_CKPT_CLASS_MARKERS = {
    "pi0": {
        "require": ["paligemma_with_expert", "action_in_proj", "time_mlp"],
        "forbid": [".layernorm.dense."],
        "forbid_hint": {
            ".layernorm.dense.": "pi0.5 AdaRMSNorm (not present in pi0)",
        },
    },
    "pi0.5": {
        "require": ["paligemma_with_expert", "action_in_proj", ".layernorm.dense."],
        "forbid": [],
        "forbid_hint": {},
    },
    "pi0_fast": {
        "require": ["paligemma_with_expert"],
        "forbid": ["action_in_proj", "action_out_proj", "time_mlp", ".layernorm.dense."],
        "forbid_hint": {
            "action_in_proj": "pi0 / pi0.5 continuous action projection (pi0-FAST decodes actions as FAST tokens, no projection)",
            "action_out_proj": "pi0 / pi0.5 continuous action projection",
            "time_mlp": "pi0 / pi0.5 flow-matching conditioning MLP",
            ".layernorm.dense.": "pi0.5 AdaRMSNorm — this ckpt is actually pi0.5",
        },
    },
}
# aliases
_CKPT_CLASS_MARKERS["pi0fast"] = _CKPT_CLASS_MARKERS["pi0_fast"]


def run_ckpt_compat_gate(policy_name: str, checkpoint: str | None) -> tuple[bool, str]:
    """Read safetensors header (no weights) to confirm checkpoint matches the
    declared policy class. Catches mislabeled ckpts before SLURM submission.

    Evidence: 2026-04-24 jadechoghari/pi0fast-libero burned three SLURM jobs
    before its pi0.5 weights were identified; header read costs one HTTP HEAD.
    """
    if not checkpoint:
        return True, "no --checkpoint — ckpt-class check skipped"
    if checkpoint.startswith("/") or Path(checkpoint).exists():
        return True, "local checkpoint — ckpt-class check skipped (trust on-disk)"
    if policy_name not in _CKPT_CLASS_MARKERS:
        return True, f"no marker profile for policy={policy_name} — ckpt-class check skipped"

    try:
        from huggingface_hub import get_safetensors_metadata
    except ImportError:
        return True, "huggingface_hub lacks get_safetensors_metadata — check skipped"

    try:
        meta = get_safetensors_metadata(checkpoint)
    except Exception as e:
        # JAX/orbax ckpts, non-safetensors repos — fall through to runtime load
        return True, f"no safetensors header ({type(e).__name__}) — check skipped"

    keys = list(meta.weight_map.keys())
    profile = _CKPT_CLASS_MARKERS[policy_name]
    issues = []
    for needle in profile["require"]:
        if not any(needle in k for k in keys):
            issues.append(f"missing tensor matching '{needle}'")
    for needle in profile["forbid"]:
        hits = [k for k in keys if needle in k]
        if hits:
            hint = profile["forbid_hint"].get(needle, "indicates a different policy class")
            issues.append(f"{len(hits)} tensors matching '{needle}' present — {hint}")
    if issues:
        lines = [f"  - {i}" for i in issues]
        return False, (
            f"checkpoint {checkpoint} does not look like {policy_name}:\n"
            + "\n".join(lines)
            + "\n  → SLURM submission would fail at load_state_dict. "
              "Pick a ckpt whose tensor keys match the policy, or change --policy."
        )
    return True, f"✓ safetensors keys match {policy_name} ({len(keys)} tensors)"


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

    # Try full benchmark name first (exact match for per-task entries like
    # "robotwin:stack_blocks_two"), then fall back to platform-only lookup.
    cfg = lookup(reg_policy, reg_bench)
    if cfg is None and ":" in benchmark_name:
        cfg = lookup(reg_policy, benchmark_name.split(":")[0])
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


def _pick_available_egl_node(nodes: list[str]) -> str:
    """Return a comma-separated nodelist of non-drained EGL nodes so SLURM
    can schedule across all of them in parallel. Earlier version returned a
    single node, which pinned every concurrent submission to the same host
    and stacked them in PENDING Reason=Resources. Why: 2026-04-23 batch of
    8 LIBERO jobs all latched onto cn26 — one ran, seven queued, while cn27
    sat mostly idle. How to apply: always return the full live-node set
    (filtering only states sinfo reports as usable).
    """
    try:
        result = subprocess.run(
            ["sinfo", "-N", "-n", ",".join(nodes), "-o", "%N %t", "--noheader"],
            capture_output=True, text=True, timeout=10,
        )
        available = []
        for line in result.stdout.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] in ("idle", "idle~", "mix", "mix-", "mixed", "alloc", "allocated"):
                available.append(parts[0])
        if available:
            return ",".join(available)
    except Exception:
        pass
    return ",".join(nodes)


def submit_as_slurm_job(policy_cfg, args, benchmark_cfg):
    """Submit the entire benchmark as a SLURM job (server + eval in one job)."""
    log_dir = args.log_dir or str(AGENT_ROOT / "logs" / "eval_results")
    job_log = str(AGENT_ROOT / "logs" / f"benchmark-{args.policy}-{args.benchmark}-%j.log")

    # When the user didn't override --port, pick a random high port per submission.
    # Multiple concurrent jobs can co-locate on the same EGL node (cn24-27), and a
    # fixed default (18800) causes all but the first to crash with
    # `OSError: [Errno 98] Address already in use`. Range avoids ephemeral ports.
    if args.port == 18800:
        args.port = random.randint(19000, 29999)
        print(f"[submit] randomized server port -> {args.port}")

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
    eval_cmd_args += _protocol_cli_extras(client_name, args.policy, args.benchmark)
    eval_args_str = " \\\n  ".join(eval_cmd_args)

    # Eval-specific env vars
    eval_env_exports = []
    for k, v in client.get("env_vars", {}).items():
        eval_env_exports.append(f"export {k}={v}")
    eval_env_block = "\n".join(eval_env_exports)

    # Robosuite-based benchmarks (libero, robocasa) need EGL-compatible nodes
    needs_egl = client_name in ("libero", "robocasa")
    if args.node:
        node_directive = f"#SBATCH --nodelist={args.node}"
    elif needs_egl and EGL_GOOD_NODES:
        egl_nodes = [n.strip() for n in EGL_GOOD_NODES.split(",") if n.strip()]
        chosen_node = _pick_available_egl_node(egl_nodes)
        node_directive = f"#SBATCH --nodelist={chosen_node}"
    else:
        node_directive = ""
    exclude_list = []
    if EGL_EXCLUDE_NODES:
        exclude_list.append(EGL_EXCLUDE_NODES)
    if client_name == "maniskill" and MANISKILL_EXCLUDE_NODES:
        exclude_list.append(MANISKILL_EXCLUDE_NODES)
    if client_name == "robotwin" and ROBOTWIN_EXCLUDE_NODES:
        exclude_list.append(ROBOTWIN_EXCLUDE_NODES)
    exclude_directive = f"#SBATCH --exclude={','.join(exclude_list)}" if exclude_list else ""

    sbatch_script = f"""#!/bin/bash
#SBATCH --job-name=bench-{args.policy[:8]}
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time={args.slurm_time}
#SBATCH --output={job_log}
{node_directive}
{exclude_directive}

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
    parser.add_argument("--slurm_time", default="02:00:00",
                        help="SLURM --time budget (HH:MM:SS); bump for high-trial RoboTwin runs")
    parser.add_argument("--list_benchmarks", action="store_true",
                        help="List all available benchmarks and exit")
    parser.add_argument("--allow_cross_domain", action="store_true",
                        help="Allow running NEEDS_FINETUNE / CROSS_DOMAIN combos (expect ~0%%)")
    parser.add_argument("--skip_preflight", action="store_true",
                        help="Skip eval_registry preflight check")
    parser.add_argument("--skip_protocol_gate", action="store_true",
                        help="Skip extract_eval_protocol --validate gate (emergency bypass; DO NOT use in normal benchmarks)")
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
                    # Empty-string value means boolean/store_true flag (e.g. --aloha).
                    # Appending "" as a separate arg confuses argparse (positional).
                    if val == "" or val is None:
                        policy_cfg.setdefault("server_args", []).append(flag)
                        print(f"  Auto-added from registry: {flag}")
                    else:
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

    # Auto-apply protocol-derived server flags (e.g. --no_flip_image,
    # --no_center_crop for openvla × ManiSkill). Runs regardless of registry hit.
    proto_server_extras = _protocol_server_extras(args.policy, args.benchmark)
    if proto_server_extras:
        flat_args = " ".join(policy_cfg.get("server_args", []))
        added = []
        for flag in proto_server_extras:
            if flag.startswith("--") and flag not in flat_args:
                policy_cfg.setdefault("server_args", []).append(flag)
                added.append(flag)
        if added:
            print(f"  Auto-added from protocol: {' '.join(added)}")

    # Final checkpoint resolution. Four layers (in priority order):
    #   1. Registry hit (handled above).
    #   2. Canonical in-domain default — checked FIRST, because HF search
    #      for "pi0_fast libero" returns 18 noisy community ckpts and the
    #      canonical `lerobot/pi0fast-libero` is buried; multiple-match branch
    #      would then exit 10 even though a canonical answer exists.
    #   3. HF variant search — auto-pick if exactly one finetuned variant;
    #      exit 10 if multiple (agent prompts user).
    #   4. Hard fail with actionable error.
    if not args.checkpoint:
        bench_short = args.benchmark.split(":")[0]  # "maniskill:PickCube-v1" -> "maniskill"
        if bench_short.startswith("libero_"):
            bench_short = "libero"  # libero_{spatial,object,goal,10,90} → libero for defaults/HF search
        query = f"{args.policy} {bench_short}"

        # Layer 2: canonical in-domain default (preempts noisy HF search)
        _LIBERO_DEFAULTS = {
            "openvla": "moojink/openvla-7b-oft-finetuned-libero-spatial",
            "pi0":     "lerobot/pi0_libero_finetuned",
            "pi0.5":   "lerobot/pi05_libero_finetuned",
            "pi0_fast": "lerobot/pi0fast-libero",
            "pi0fast":  "lerobot/pi0fast-libero",
            "smolvla": "lerobot/smolvla_base",
            "spatialvla": "IPEC-COMMUNITY/spatialvla-4b-224-pt",
        }
        if bench_short == "libero" and args.policy in _LIBERO_DEFAULTS:
            args.checkpoint = _LIBERO_DEFAULTS[args.policy]
            print(f"  No explicit --checkpoint — using canonical LIBERO ckpt: {args.checkpoint}")
        else:
            # Layer 3: HF variant search via download_model
            print(f"  No checkpoint provided — searching HF for '{query}' ...")
            DM = Path(__file__).parent.parent / "download_model" / "download_model.py"
            proc = subprocess.run(
                [sys.executable, str(DM), "--repo-id", query, "--list-only", "true"],
                capture_output=True, text=True, timeout=60,
            )
            try:
                variants = json.loads(proc.stdout[proc.stdout.find("{"):proc.stdout.rfind("}")+1])
                cands = variants.get("variants", [])
            except Exception:
                cands = []
            _FT_PAT = re.compile(r"finetune|finetuned|\bft\b|\bsft\b|lora|\bdpo\b|grpo|rlhf|pretrain", re.IGNORECASE)
            ft = [c for c in cands if _FT_PAT.search(c)]
            if len(ft) == 1:
                args.checkpoint = ft[0]
                print(f"  Auto-picked single finetuned variant: {args.checkpoint}")
            elif len(ft) > 1:
                print(json.dumps({
                    "search_term": query,
                    "variants": ft + [c for c in cands if c not in ft][:10],
                    "preferred": "(multiple finetuned matches — pick one)",
                }, indent=2))
                print(f"\n  Multiple finetuned matches for {query} — pass --checkpoint <repo_id> or let the agent disambiguate.")
                sys.exit(10)
            else:
                # Layer 4: no ckpt. Previous behaviour silently cross-domain
                # fell back to a LIBERO ckpt (e.g. pi0 × maniskill →
                # lerobot/pi0_libero_finetuned with a warning), which
                # guaranteed ~0% and burned an H100 job. User directive
                # 2026-04-24: "只跑有对应权证的" — hard fail with actionable error.
                print(
                    f"\nERROR: No finetuned {args.policy} checkpoint found for "
                    f"benchmark '{args.benchmark}'.\n"
                    f"  • registry miss, HF search '{query}' returned no finetuned variant,\n"
                    f"  • refusing cross-domain fallback (would burn compute for ~0% results).\n"
                    f"  Fix: pass --checkpoint <repo_id> for a ckpt actually trained on\n"
                    f"  {args.benchmark}, or train one first."
                )
                sys.exit(1)

    # Capture user-facing policy name BEFORE openpi routing rewrite — the
    # protocol gate and registry lookups must use the original name the user
    # typed (e.g. pi0.5 → pi0.5_maniskill.json, not openpi_maniskill.json).
    user_policy_name = args.policy

    # Resolve lerobot-training nested checkpoint layout (checkpoints/<step>/pretrained_model/).
    # Must run before openpi detection — the nested dir is where openpi-layout
    # sentinels live too, and we'd rather the lerobot server load it directly.
    if (args.policy in ("pi0", "pi0.5", "pi0_fast", "pi0fast", "smolvla", "lerobot")
            and args.checkpoint
            and not args.checkpoint.startswith(("hf://", "lerobot/"))):
        args.checkpoint = resolve_lerobot_nested_ckpt(args.checkpoint)

    # Auto-route to OpenPI server if checkpoint is in JAX/orbax format. RLinf,
    # kimtaey, SakikoTogawa community checkpoints all use this — they can't be
    # loaded by the lerobot/pi0 PyTorch server. The openpi/scripts/policy_server.py
    # handles them. We pick the closest stock --config based on the benchmark.
    if (args.policy in ("pi0", "pi0.5", "pi0_fast", "pi0fast", "lerobot")
            and args.checkpoint
            and not args.checkpoint.startswith(("hf://", "lerobot/"))
            and is_openpi_checkpoint(args.checkpoint)):
        openpi_cfg_name = route_openpi_config(args.benchmark, args.checkpoint)
        print(f"  [route] Detected OpenPI/orbax format → switching to openpi server")
        print(f"          --config {openpi_cfg_name} (matched to '{args.benchmark}')")

        # Patch norm_stats path: stock openpi configs look at
        # assets/<asset_id>/norm_stats.json (e.g. physical-intelligence/libero),
        # but community ckpts use custom asset_ids ("robocasa_lerobot_30demos"
        # for kimtaey, "<bm>" under physical-intelligence/ for RLinf HF mirror).
        # Symlink the actual norm_stats to the path the chosen config expects.
        try:
            ckpt_p = Path(args.checkpoint)
            asset_target = ckpt_p / "assets" / "physical-intelligence" / openpi_cfg_name.replace("pi0_fast_", "").replace("pi05_", "").replace("pi0_", "")
            # For pi0_fast_libero → "libero"; pi0_fast_droid → "droid"
            target_path = asset_target / "norm_stats.json"
            if not target_path.exists():
                # Find existing norm_stats.json anywhere under this ckpt
                candidates = list(ckpt_p.rglob("norm_stats.json"))
                if candidates:
                    asset_target.mkdir(parents=True, exist_ok=True)
                    target_path.symlink_to(candidates[0])
                    print(f"  [patch] symlinked norm_stats: {candidates[0]} → {target_path}")
        except Exception as e:
            print(f"  [warn] norm_stats auto-patch failed: {e}")

        args.policy = "openpi"
        policy_cfg = resolve_policy("openpi")
        # Substitute openpi_config into server_args template
        policy_cfg = dict(policy_cfg)
        policy_cfg["server_args"] = [
            a.replace("{openpi_config}", openpi_cfg_name) for a in policy_cfg["server_args"]
        ]

    print(f"\n=== Benchmark Run ===")
    print(f"Policy:     {args.policy}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Benchmark:  {args.benchmark}")
    print(f"Platform:   {client_name}")
    print(f"Task ID:    {benchmark_cfg['task_id']}")
    print(f"Trials:     {args.num_trials}")

    # Protocol gate: force eval_protocol.json to exist + be fully cited before
    # burning compute. PullCube 40% vs paper 92% came from silently missing
    # center_crop=False — this gate catches that class of bug before submission.
    if not args.skip_protocol_gate:
        ok, msg = run_protocol_gate(user_policy_name, args.benchmark, args.checkpoint)
        print(f"\n[protocol_gate] {msg}")
        if not ok:
            print("\n❌ Protocol gate failed — refusing to submit.")
            print("   Fix: run `extract_eval_protocol` and cite each field from the author's")
            print("   paper/README/eval-repo, then re-run. To bypass (not recommended): --skip_protocol_gate")
            sys.exit(4)

        # Ckpt-class compatibility sensor (reads safetensors header, no weights)
        ok2, msg2 = run_ckpt_compat_gate(user_policy_name, args.checkpoint)
        print(f"[ckpt_compat]   {msg2}")
        if not ok2:
            print("\n❌ Checkpoint class mismatch — refusing to submit.")
            print("   Fix: pick a checkpoint whose safetensors keys match the --policy class.")
            print("   Bypass (not recommended): --skip_protocol_gate")
            sys.exit(5)

    # Mode 1: Submit as SLURM job
    has_slurm = subprocess.run(["which", "sbatch"], capture_output=True).returncode == 0
    try:
        has_gpu = subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except FileNotFoundError:
        has_gpu = False
    if not args.submit and has_slurm and not has_gpu and not args.server_addr:
        args.submit = True
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
