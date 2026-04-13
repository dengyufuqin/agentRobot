#!/usr/bin/env python3
"""Validate a generated policy_server.py adapter in a sandboxed subprocess.

Runs three escalating checks against an adapter file:
  1. syntax  — `py_compile` of the adapter (instant)
  2. import  — import the module under the repo's venv (catches ImportError)
  3. smoke   — instantiate Policy() and call infer() with a fake obs (catches API mismatches)

Each check that fails returns a STRUCTURED report so the LLM-agent can read
the traceback and decide what to fix. We deliberately do NOT load real model
weights — that takes 60-120s on GPU and the goal here is fast iteration.

Usage:
    validate.py <repo_path> --python <interp> --pythonpath <p1:p2> --mode <syntax|import|smoke>
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("repo_path", help="Path to repo containing policy_server.py")
parser.add_argument("--python", default=sys.executable,
                    help="Python interpreter to validate against (use repo's venv python)")
parser.add_argument("--pythonpath", default="",
                    help="Colon-separated PYTHONPATH entries")
parser.add_argument("--mode", choices=["syntax", "import", "smoke"], default="import",
                    help="syntax = py_compile only; import = also import the module; "
                         "smoke = also call Policy().infer() with a fake observation")
parser.add_argument("--adapter", default="policy_server.py",
                    help="Adapter filename (default: policy_server.py)")
parser.add_argument("--policy-class", default=None,
                    help="(smoke mode) Policy class name to instantiate. "
                         "If omitted, the validator picks the first BasePolicy subclass.")
parser.add_argument("--timeout", type=int, default=60)
args = parser.parse_args()

repo = Path(args.repo_path).expanduser().resolve()
if not repo.exists():
    print(f"[ERROR] repo path does not exist: {repo}")
    sys.exit(2)

adapter_path = repo / args.adapter
if not adapter_path.exists():
    print(f"[ERROR] adapter file not found: {adapter_path}")
    sys.exit(2)

print(f"[VALIDATE] adapter={adapter_path}")
print(f"[VALIDATE] python={args.python}")
print(f"[VALIDATE] mode={args.mode}")
print("---")


# ---------- 1. syntax check ----------
print("[1/?] syntax check (py_compile)...")
syn = subprocess.run(
    [args.python, "-m", "py_compile", str(adapter_path)],
    capture_output=True, text=True, timeout=args.timeout,
)
if syn.returncode != 0:
    print("FAIL: syntax error")
    print(syn.stderr.strip())
    sys.exit(1)
print("OK: syntax")

if args.mode == "syntax":
    print("\n[DONE] syntax-only mode")
    sys.exit(0)


# ---------- 2. import check ----------
print("[2/?] import check (load module under repo venv)...")

# We need an env that includes the repo and any user-provided pythonpath
env = os.environ.copy()
pp_parts = []
if args.pythonpath:
    pp_parts.extend(args.pythonpath.split(":"))
pp_parts.append(str(repo))
if env.get("PYTHONPATH"):
    pp_parts.append(env["PYTHONPATH"])
env["PYTHONPATH"] = ":".join([p for p in pp_parts if p])

# Build a tiny driver script that imports the adapter as a module and reports
# what it found. We run it in a subprocess so any segfaults / huge imports
# are isolated.
driver = r"""
import importlib.util, inspect, json, sys, traceback
from pathlib import Path

ADAPTER = Path(sys.argv[1])
try:
    spec = importlib.util.spec_from_file_location("agent_adapter", str(ADAPTER))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
except BaseException as e:
    print("IMPORT_FAIL")
    traceback.print_exc()
    sys.exit(1)

# Discover BasePolicy subclasses
classes = []
try:
    from policy_websocket import BasePolicy
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj is not BasePolicy and issubclass(obj, BasePolicy) and obj.__module__ == mod.__name__:
            classes.append(name)
except Exception:
    # policy_websocket not importable — fall back to scanning module attrs
    for name, obj in inspect.getmembers(mod, inspect.isclass):
        if obj.__module__ == mod.__name__ and "Policy" in name:
            classes.append(name)

print("IMPORT_OK")
print("CLASSES=" + json.dumps(classes))
"""

imp = subprocess.run(
    [args.python, "-c", driver, str(adapter_path)],
    capture_output=True, text=True, timeout=args.timeout, env=env,
)
print(imp.stdout.strip())
if imp.returncode != 0:
    print("FAIL: import error")
    if imp.stderr:
        print("[STDERR]")
        print(imp.stderr.strip())
    sys.exit(1)
print("OK: import")

# Parse classes from driver output
classes = []
for line in imp.stdout.splitlines():
    if line.startswith("CLASSES="):
        import json as _j
        try:
            classes = _j.loads(line[len("CLASSES="):])
        except Exception:
            classes = []
print(f"  found policy classes: {classes}")

if not classes:
    print("WARN: no BasePolicy subclass found in adapter")

if args.mode == "import":
    print("\n[DONE] import-only mode")
    sys.exit(0)


# ---------- 3. smoke test ----------
print("[3/3] smoke test (instantiate + fake infer)...")
policy_class = args.policy_class or (classes[0] if classes else None)
if not policy_class:
    print("FAIL: smoke test needs --policy-class (no BasePolicy subclass auto-detected)")
    sys.exit(1)

smoke = r"""
import importlib.util, sys, traceback
import numpy as np
from pathlib import Path

ADAPTER = Path(sys.argv[1])
CLASS_NAME = sys.argv[2]

spec = importlib.util.spec_from_file_location("agent_adapter", str(ADAPTER))
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except BaseException:
    print("SMOKE_FAIL: import")
    traceback.print_exc()
    sys.exit(1)

cls = getattr(mod, CLASS_NAME, None)
if cls is None:
    print(f"SMOKE_FAIL: class {CLASS_NAME} not found")
    sys.exit(1)

# Instantiate with a no-op checkpoint — we DON'T want to actually load weights
try:
    policy = cls(checkpoint="__validate_only__", _validate_only=True)
except TypeError:
    # Adapter doesn't accept _validate_only kwarg — try with just checkpoint
    try:
        policy = cls(checkpoint="__validate_only__")
    except BaseException:
        print("SMOKE_FAIL: __init__ raised — adapter must support a no-op init or _validate_only kwarg")
        traceback.print_exc()
        sys.exit(1)
except BaseException:
    print("SMOKE_FAIL: __init__ raised")
    traceback.print_exc()
    sys.exit(1)

# Send a fake "init" call (no images) — every adapter must handle this
try:
    out = policy.infer({"action_dim": 7})
    assert isinstance(out, dict), f"infer() must return dict, got {type(out)}"
    assert "actions" in out, f"infer() must return {{'actions': ...}}, got keys {list(out.keys())}"
    arr = out["actions"]
    assert hasattr(arr, "__len__"), "actions must be array-like"
    print(f"SMOKE_OK: infer(no-image) returned actions of len={len(arr)}")
except BaseException:
    print("SMOKE_FAIL: infer({}) raised")
    traceback.print_exc()
    sys.exit(1)

sys.exit(0)
"""
sk = subprocess.run(
    [args.python, "-c", smoke, str(adapter_path), policy_class],
    capture_output=True, text=True, timeout=args.timeout, env=env,
)
print(sk.stdout.strip())
if sk.returncode != 0:
    print("FAIL: smoke test")
    if sk.stderr:
        print("[STDERR]")
        print(sk.stderr.strip())
    sys.exit(1)

print("\n[DONE] all checks passed")
