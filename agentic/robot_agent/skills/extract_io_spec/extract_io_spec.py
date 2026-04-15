#!/usr/bin/env python3
"""Turn probe_run's .probe_io_spec.json into a human-readable obs/action spec summary.

Given the JSON emitted by probe_run's --io-spec-hook, produce:
  - image keys + shapes + dtypes
  - state/proprio keys + shapes
  - action head output shape + dim
  - a best-guess obs/action dict template the adapter can use
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _walk_tensors(node, prefix=""):
    """Yield (path, {shape,dtype,finite,device}) for every tensor-like dict."""
    if isinstance(node, dict):
        if "shape" in node and "dtype" in node:
            yield prefix, node
            return
        for k, v in node.items():
            yield from _walk_tensors(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk_tensors(v, f"{prefix}[{i}]")


def classify(path: str, meta: dict) -> str:
    """Guess tensor category from path + shape."""
    shape = meta.get("shape", [])
    lname = path.lower()
    if any(k in lname for k in ["image", "img", "rgb", "pixel", "cam"]):
        return "image"
    if any(k in lname for k in ["state", "proprio", "qpos", "joint"]):
        return "state"
    if any(k in lname for k in ["action", "target"]) or (len(shape) == 2 and shape[-1] <= 16):
        return "action_like"
    # Shape-based heuristics
    if len(shape) == 4 and shape[-1] >= 64 and shape[-2] >= 64:
        return "image"
    if len(shape) == 2 and shape[-1] <= 32:
        return "state"
    return "other"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec-file", required=True,
                    help="Path to .probe_io_spec.json from probe_run")
    ap.add_argument("--out", default="",
                    help="Optional path to dump structured JSON summary")
    args = ap.parse_args()

    p = Path(args.spec_file)
    if not p.is_file():
        print(f"ERROR: spec file not found: {p}")
        return 1

    try:
        data = json.loads(p.read_text())
    except Exception as e:
        print(f"ERROR: could not parse JSON: {e}")
        return 1

    if not isinstance(data, list) or not data:
        print(f"ERROR: expected non-empty list of calls, got {type(data).__name__}")
        return 1

    summary: dict = {"calls": []}
    for i, call in enumerate(data):
        module = call.get("module_class", "?")
        inputs = {}
        for arg_idx, arg in enumerate(call.get("input_args", [])):
            for path, meta in _walk_tensors(arg, f"arg[{arg_idx}]"):
                inputs[path] = {**meta, "category": classify(path, meta)}
        for k, v in call.get("input_kwargs", {}).items():
            for path, meta in _walk_tensors(v, k):
                inputs[path] = {**meta, "category": classify(path, meta)}
        out_info = {}
        for path, meta in _walk_tensors(call.get("output"), "output"):
            out_info[path] = {**meta, "category": classify(path, meta)}
        summary["calls"].append({
            "i": i,
            "module": module,
            "inputs": inputs,
            "outputs": out_info,
        })

    # Derived spec from call 0
    first = summary["calls"][0]
    images = [k for k, v in first["inputs"].items() if v["category"] == "image"]
    states = [k for k, v in first["inputs"].items() if v["category"] == "state"]
    actions = [k for k, v in first["outputs"].items() if v["category"] == "action_like"]

    # Pick primary action — first output
    primary_action = None
    if actions:
        primary_action = first["outputs"][actions[0]]
    elif first["outputs"]:
        primary_action = next(iter(first["outputs"].values()))

    summary["derived"] = {
        "image_keys": images,
        "state_keys": states,
        "action_shape": primary_action["shape"] if primary_action else None,
        "action_dim": (primary_action["shape"][-1] if primary_action and primary_action["shape"] else None),
    }

    out_json = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).write_text(out_json)
    print(out_json)

    # Human summary on stderr
    print("\n=== IO SPEC SUMMARY ===", file=sys.stderr)
    print(f"Module: {first['module']}", file=sys.stderr)
    print(f"Images: {images}", file=sys.stderr)
    for k in images:
        print(f"  {k}: {first['inputs'][k]['shape']} {first['inputs'][k]['dtype']}", file=sys.stderr)
    print(f"States: {states}", file=sys.stderr)
    for k in states:
        print(f"  {k}: {first['inputs'][k]['shape']} {first['inputs'][k]['dtype']}", file=sys.stderr)
    if primary_action:
        print(f"Action: shape={primary_action['shape']} dtype={primary_action['dtype']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
