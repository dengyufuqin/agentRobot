#!/usr/bin/env python3
"""Probe a benchmark/simulator repo: verify it imports, enumerate tasks/suites,
capture obs/action space, pick a default task.

Symmetric to probe_run (which handles POLICY repos) — this handles SIMULATOR
repos: LIBERO, ManiSkill, RoboCasa, CALVIN, SimplerEnv, RoboTwin, etc.

Strategy:
  1. Try a list of known env factories (gym.make, env = Env(), suite.Suite(), ...)
  2. For each factory that imports, reset it, serialize obs+action_space.
  3. Return the first working (factory, task_id) pair + obs/action schema.

Output JSON:
{
  "benchmark": "libero",
  "import_ok": true,
  "factory": "libero.envs.LiberoEnv",
  "sample_task": "libero_spatial_pick_up_the_black_bowl",
  "obs_keys": ["agentview_image", "robot0_eef_pos", "robot0_joint_pos", ...],
  "obs_shapes": {"agentview_image": [256, 256, 3], ...},
  "action_shape": [7],
  "action_low": [-1,...], "action_high": [1,...]
}

Exit 0 = importable + at least one sample task works
Exit 2 = import fails
Exit 3 = imports but no factory works
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path


# `LITE_PROBES`: import + enumerate, no env.reset(). Runs anywhere — no GPU/EGL.
# `PROBES`: full reset → captures real obs/action shape. Needs GPU for MUJOCO_GL=egl.
# Always tries LITE first — if that fails the repo is fundamentally broken.
LITE_PROBES: dict[str, str] = {
    "libero": r"""
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
suites = list(d.keys())
suite = d[suites[0]]()
n_tasks = suite.n_tasks if hasattr(suite, "n_tasks") else len(getattr(suite, "tasks", [])) or 10
sample_task = suite.get_task(0)
result = {
    "benchmark": "libero",
    "mode": "lite",
    "factory": "libero.libero.envs.OffScreenRenderEnv",
    "suites": suites,
    "sample_suite": suites[0],
    "sample_task": getattr(sample_task, "name", str(sample_task)),
    "n_tasks_in_suite": n_tasks,
}
""",
    "maniskill": r"""
import gymnasium as gym
import mani_skill.envs  # registers ManiSkill envs into gym registry
# ManiSkill envs are tagged with a version suffix and one of these keywords
TAGS = ("Cube", "Pick", "Push", "Stack", "Peg", "Open", "Close",
        "Place", "Plug", "Insert", "Rotate", "Lift")
ms_envs = sorted([k for k in gym.envs.registry.keys()
                  if any(t in k for t in TAGS) and "-v" in k])
result = {
    "benchmark": "maniskill",
    "mode": "lite",
    "factory": "gymnasium.make",
    "sample_envs": ms_envs[:8],
    "n_envs": len(ms_envs),
}
""",
    "robocasa": r"""
import robocasa
result = {"benchmark": "robocasa", "mode": "lite",
          "version": getattr(robocasa, "__version__", "?")}
""",
    "calvin": r"""
import calvin_env
result = {"benchmark": "calvin", "mode": "lite", "import_ok": True}
""",
    "simpler": r"""
import simpler_env
result = {"benchmark": "simpler", "mode": "lite",
          "n_envs": len(getattr(simpler_env, "ENVIRONMENTS", []))}
""",
}

# (benchmark_id, python probe snippet) — full env.reset()
PROBES: dict[str, str] = {
    "libero": r"""
import os, json, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
benchmark_dict = benchmark.get_benchmark_dict()
suite_name = next(iter(benchmark_dict.keys()))
suite = benchmark_dict[suite_name]()
task = suite.get_task(0)
env_args = {"bddl_file_name": suite.get_task_bddl_file_path(0),
            "camera_heights": 128, "camera_widths": 128}
env = OffScreenRenderEnv(**env_args)
obs = env.reset()
result = {
    "benchmark": "libero",
    "factory": "libero.libero.envs.OffScreenRenderEnv",
    "sample_suite": suite_name,
    "sample_task": task.name if hasattr(task, 'name') else str(task),
    "obs_keys": list(obs.keys()) if isinstance(obs, dict) else ["<array>"],
    "obs_shapes": {k: list(np.asarray(v).shape) for k, v in obs.items()} if isinstance(obs, dict) else None,
    "action_shape": list(env.action_space.shape) if hasattr(env, "action_space") else None,
}
env.close()
""",
    "maniskill": r"""
import os, json, numpy as np, gymnasium as gym
os.environ.setdefault("MUJOCO_GL", "egl")
import mani_skill.envs  # registers envs
env_id = "PickCube-v1"
env = gym.make(env_id, obs_mode="rgbd", control_mode="pd_joint_delta_pos")
obs, info = env.reset(seed=0)
def _walk(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items(): out.update(_walk(v, f"{prefix}{k}/"))
    else:
        try: out[prefix[:-1]] = list(np.asarray(d).shape)
        except Exception: out[prefix[:-1]] = "<non-array>"
    return out
result = {
    "benchmark": "maniskill",
    "factory": "gymnasium.make",
    "sample_task": env_id,
    "obs_shapes": _walk(obs),
    "action_shape": list(env.action_space.shape) if hasattr(env, "action_space") else None,
}
env.close()
""",
    "robocasa": r"""
import os, json, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import robocasa
from robomimic.envs.env_robosuite import EnvRobosuite
task = "PnPCounterToCab"
env = EnvRobosuite(env_name=task, has_renderer=False, has_offscreen_renderer=True,
                   use_camera_obs=True, camera_names=["robot0_agentview_left"])
obs = env.reset()
result = {
    "benchmark": "robocasa",
    "factory": "robomimic.envs.env_robosuite.EnvRobosuite",
    "sample_task": task,
    "obs_keys": list(obs.keys()) if isinstance(obs, dict) else ["<array>"],
    "obs_shapes": {k: list(np.asarray(v).shape) for k, v in obs.items()} if isinstance(obs, dict) else None,
    "action_shape": list(env.env.action_spec[0].shape),
}
""",
    "calvin": r"""
import os, json, numpy as np, hydra
os.environ.setdefault("MUJOCO_GL", "egl")
from calvin_env.envs.play_table_env import PlayTableSimEnv
env = PlayTableSimEnv()  # defaults
obs = env.reset()
result = {
    "benchmark": "calvin",
    "factory": "calvin_env.envs.play_table_env.PlayTableSimEnv",
    "obs_keys": list(obs.keys()) if isinstance(obs, dict) else ["<array>"],
    "obs_shapes": {k: list(np.asarray(v).shape) for k, v in obs.items()} if isinstance(obs, dict) else None,
    "action_shape": list(env.action_space.shape) if hasattr(env, "action_space") else None,
}
""",
    "simpler": r"""
import os, json, numpy as np
os.environ.setdefault("MUJOCO_GL", "egl")
import simpler_env
env_id = simpler_env.ENVIRONMENTS[0]
env = simpler_env.make(env_id)
obs, reset_info = env.reset()
result = {
    "benchmark": "simpler",
    "factory": "simpler_env.make",
    "sample_task": env_id,
    "obs_keys": list(obs.keys()) if isinstance(obs, dict) else ["<array>"],
    "action_shape": list(env.action_space.shape) if hasattr(env, "action_space") else None,
}
""",
}


def build_probe_script(body: str) -> str:
    return textwrap.dedent(f'''
    import json, sys, traceback
    try:
{textwrap.indent(body.strip(), "        ")}
        print("=== PROBE_OK ===")
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print("=== PROBE_FAIL ===")
        print(f"{{type(e).__name__}}: {{e}}")
        traceback.print_exc()
        sys.exit(3)
    ''').lstrip()


def run_one(repo_path: Path, venv_python: Path, benchmark: str, body: str,
            timeout: int) -> tuple[int, dict | None, str]:
    script_path = repo_path / f".probe_benchmark_{benchmark}.py"
    script_path.write_text(build_probe_script(body))
    try:
        proc = subprocess.run(
            [str(venv_python), "-u", str(script_path)],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return 2, None, f"TIMEOUT after {timeout}s"
    finally:
        try: script_path.unlink()
        except Exception: pass

    out = proc.stdout + proc.stderr
    if proc.returncode == 0 and "=== PROBE_OK ===" in out:
        try:
            payload = out.split("=== PROBE_OK ===", 1)[1].strip()
            # Handle trailing text after JSON
            brace = payload.find("{")
            last = payload.rfind("}")
            return 0, json.loads(payload[brace:last + 1]), out
        except Exception as e:
            return 3, None, f"parse failed: {e}\n---\n{out[-500:]}"
    return max(proc.returncode, 2), None, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", required=True, type=Path)
    ap.add_argument("--venv-python", type=Path, default=None)
    ap.add_argument("--benchmark", default="",
                    help="Explicit benchmark id (libero, maniskill, robocasa, calvin, simpler). "
                         "If empty, try all known probes.")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--lite", action="store_true",
                    help="Skip env.reset() — only verify import + enumerate tasks. Runs without GPU.")
    ap.add_argument("--out", default="", help="Optional output path for JSON")
    args = ap.parse_args()

    repo = args.repo_path.resolve()
    if not repo.is_dir():
        print(f"ERROR: repo not found: {repo}")
        return 1
    venv_py = args.venv_python or (repo / ".venv" / "bin" / "python3")
    if not venv_py.is_file() or not os.access(venv_py, os.X_OK):
        print(f"ERROR: venv python missing: {venv_py}")
        return 1

    probe_set = LITE_PROBES if args.lite else PROBES
    mode_label = "lite" if args.lite else "full"
    candidates = [args.benchmark] if args.benchmark else list(probe_set.keys())
    tried: list[dict] = []
    winner: dict | None = None

    for bm in candidates:
        if bm not in probe_set:
            tried.append({"benchmark": bm, "status": "unknown probe"})
            continue
        print(f"[ONBOARD] trying {bm} ({mode_label}) ...", flush=True)
        rc, payload, out = run_one(repo, venv_py, bm, probe_set[bm], args.timeout)
        if rc == 0 and payload:
            print(f"[ONBOARD] ✓ {bm} works ({mode_label})")
            winner = payload
            tried.append({"benchmark": bm, "status": "ok"})
            break
        else:
            tail = out.strip().splitlines()[-5:] if out else []
            print(f"[ONBOARD] ✗ {bm} failed (rc={rc})")
            tried.append({"benchmark": bm, "status": "fail", "error_tail": tail})

    report = {"winner": winner, "tried": tried, "repo": str(repo)}
    out_s = json.dumps(report, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(out_s)
    print(out_s)
    return 0 if winner else 3


if __name__ == "__main__":
    sys.exit(main())
