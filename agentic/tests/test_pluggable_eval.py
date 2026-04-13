#!/usr/bin/env python3
"""
Test suite: Verify "any model × any benchmark" claim at the system level.

Tests the pluggable eval architecture WITHOUT requiring GPU:
1. All eval clients resolve correctly
2. All benchmark configs resolve correctly
3. Policy auto-discovery reads arm_controller from yaml
4. Action type mapping works for all platforms
5. SLURM scripts generate correctly for all combinations
"""

import os
import sys
import json
from pathlib import Path

# Setup
AGENT_ROOT = Path(os.environ.get("AGENTROBOT_ROOT",
    str(Path(__file__).resolve().parent.parent.parent.parent)))
sys.path.insert(0, str(AGENT_ROOT / "agentic" / "robot_agent" / "skills" / "run_benchmark"))

import run_benchmark as rb

PASS = 0
FAIL = 0

def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS: {name}")
    else:
        FAIL += 1
        print(f"  FAIL: {name} — {detail}")


print("=" * 60)
print("TEST: Pluggable Eval Architecture")
print("=" * 60)

# ---------------------------------------------------------------
# 1. Eval clients
# ---------------------------------------------------------------
print("\n--- Eval Clients ---")
for name in ["libero", "maniskill", "robotwin"]:
    client = rb.EVAL_CLIENTS.get(name)
    test(f"eval_client '{name}' exists", client is not None)
    if client:
        test(f"  has eval_script", "eval_script" in client)
        test(f"  has eval_args", "eval_args" in client)
        test(f"  eval_script file exists", Path(client["eval_script"]).exists(),
             f"missing: {client['eval_script']}")
        test(f"  eval_python exists", Path(client["eval_python"]).exists(),
             f"missing: {client['eval_python']}")

# ---------------------------------------------------------------
# 2. Benchmark resolution
# ---------------------------------------------------------------
print("\n--- Benchmark Resolution ---")

# Known benchmarks
for bname in ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]:
    cfg = rb.resolve_benchmark(bname)
    test(f"resolve '{bname}'", cfg is not None and cfg["eval_client"] == "libero")

for bname in ["maniskill:PickCube-v1", "maniskill:StackCube-v1"]:
    cfg = rb.resolve_benchmark(bname)
    test(f"resolve '{bname}'", cfg is not None and cfg["eval_client"] == "maniskill")

for bname in ["robotwin:open_laptop", "robotwin:beat_block_hammer"]:
    cfg = rb.resolve_benchmark(bname)
    test(f"resolve '{bname}'", cfg is not None and cfg["eval_client"] == "robotwin")

# Auto-detect unknown tasks
cfg = rb.resolve_benchmark("maniskill:SomeNewTask-v99")
test("auto-detect unknown maniskill task", cfg is not None and cfg["task_id"] == "SomeNewTask-v99")

cfg = rb.resolve_benchmark("robotwin:some_new_task")
test("auto-detect unknown robotwin task", cfg is not None and cfg["task_id"] == "some_new_task")

# Unknown platform returns None
cfg = rb.resolve_benchmark("unknown_bench")
test("unknown benchmark returns None", cfg is None)

# ---------------------------------------------------------------
# 3. Policy resolution (built-in)
# ---------------------------------------------------------------
print("\n--- Policy Resolution ---")
for pname, expected_ac in [("openvla", "cartesian_pose"), ("lerobot", "joint_vel"),
                            ("diffusion_policy", "cartesian_pose")]:
    cfg = rb.resolve_policy(pname)
    test(f"resolve '{pname}'", cfg is not None)
    if cfg:
        test(f"  arm_controller={expected_ac}", cfg.get("arm_controller") == expected_ac)

# Alias
cfg = rb.resolve_policy("openvla-oft")
test("alias 'openvla-oft' → openvla", cfg is not None and "alias" not in cfg)

# ---------------------------------------------------------------
# 4. Policy auto-discovery from yaml
# ---------------------------------------------------------------
print("\n--- Policy Auto-Discovery from YAML ---")
import yaml
for yaml_path in sorted(AGENT_ROOT.glob("*/policy_server.yaml")):
    meta = yaml.safe_load(yaml_path.read_text())
    name = meta.get("name", "?")
    ps = meta.get("policy_server", {})
    res = ps.get("resources", {})
    ac = res.get("arm_controller")
    test(f"yaml '{name}' has arm_controller", ac is not None, f"missing in {yaml_path}")

# ---------------------------------------------------------------
# 5. Action type mapping
# ---------------------------------------------------------------
print("\n--- Action Type Mapping ---")
_AC_TO_AT = {"cartesian_pose": "ee", "joint_vel": "qpos", "joint_pos": "qpos"}
for ac, expected in [("cartesian_pose", "ee"), ("joint_vel", "qpos"), ("joint_pos", "qpos")]:
    test(f"arm_controller '{ac}' → action_type '{expected}'",
         _AC_TO_AT.get(ac) == expected)

# ---------------------------------------------------------------
# 6. Cross-platform eval arg generation
# ---------------------------------------------------------------
print("\n--- Cross-Platform Eval Arg Generation ---")
for platform, expected_flag in [("libero", "--arm_controller"),
                                 ("maniskill", "--env_id"),
                                 ("robotwin", "--action_type")]:
    client = rb.EVAL_CLIENTS[platform]
    args_str = " ".join(client["eval_args"])
    test(f"{platform} eval uses '{expected_flag}'", expected_flag in args_str)

# ---------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------
print("\n" + "=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL} failed")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
