#!/usr/bin/env python3
"""Validate a dataloader factory: pull 1-2 batches, report keys/shapes/dtypes/finite.

Purpose: before launching a multi-hour finetune, prove that:
  1. The factory function imports cleanly.
  2. It can be instantiated (dataset exists, paths resolve).
  3. At least one batch comes out.
  4. Tensors are finite, have reasonable shapes, and expected keys are present.

Exit codes:
  0  all requested checks passed
  1  bad args / factory not importable
  2  factory raised on instantiation
  3  first batch raised / timeout
  4  batch missing expected keys OR contains NaN/Inf OR shape mismatch
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path


def _describe(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            finite = bool(torch.isfinite(x).all().item()) if x.is_floating_point() else True
            return {
                "shape": list(x.shape),
                "dtype": str(x.dtype),
                "finite": finite,
                "device": str(x.device),
            }
    except ImportError:
        pass
    try:
        import numpy as np
        if isinstance(x, np.ndarray):
            finite = bool(np.isfinite(x).all()) if np.issubdtype(x.dtype, np.floating) else True
            return {"shape": list(x.shape), "dtype": str(x.dtype), "finite": finite}
    except ImportError:
        pass
    if isinstance(x, (list, tuple)):
        return [_describe(v) for v in x[:3]]
    if isinstance(x, dict):
        return {k: _describe(v) for k, v in x.items()}
    return {"type": type(x).__name__, "repr": repr(x)[:80]}


def load_factory(module_path: str, func_name: str):
    """Import a module by dotted path OR file path, return named callable."""
    if module_path.endswith(".py") or "/" in module_path:
        p = Path(module_path).resolve()
        spec = importlib.util.spec_from_file_location("_dl_factory", p)
        if spec is None or spec.loader is None:
            raise ImportError(f"could not build spec for {p}")
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(p.parent))
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(module_path)
    if not hasattr(mod, func_name):
        raise AttributeError(f"{module_path} has no attribute '{func_name}'")
    return getattr(mod, func_name)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory-module", required=True,
                    help="Dotted import path OR .py file path containing the factory")
    ap.add_argument("--factory-func", default="make_dataloader",
                    help="Function name inside the module")
    ap.add_argument("--factory-kwargs", default="{}",
                    help="JSON dict of kwargs to pass to the factory")
    ap.add_argument("--expected-keys", default="",
                    help="Comma-separated keys that must appear in each batch (e.g. 'image,state,action')")
    ap.add_argument("--num-batches", type=int, default=2,
                    help="How many batches to pull")
    ap.add_argument("--pythonpath", default="",
                    help="Colon-separated extra PYTHONPATH entries")
    ap.add_argument("--out", default="",
                    help="Optional path to dump the report as JSON")
    args = ap.parse_args()

    if args.pythonpath:
        for p in args.pythonpath.split(":"):
            if p:
                sys.path.insert(0, p)

    try:
        kwargs = json.loads(args.factory_kwargs)
        if not isinstance(kwargs, dict):
            raise ValueError("factory-kwargs must be a JSON object")
    except Exception as e:
        print(f"ERROR: bad --factory-kwargs: {e}")
        return 1

    report = {
        "factory": f"{args.factory_module}:{args.factory_func}",
        "factory_kwargs": kwargs,
        "stages": {},
    }

    # Stage 1: import
    t0 = time.time()
    try:
        factory = load_factory(args.factory_module, args.factory_func)
        report["stages"]["import"] = {"ok": True, "seconds": round(time.time() - t0, 2)}
    except Exception as e:
        report["stages"]["import"] = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
        _emit(report, args.out)
        print(f"IMPORT FAIL: {e}", file=sys.stderr)
        return 1

    # Stage 2: instantiate
    t0 = time.time()
    try:
        dl = factory(**kwargs)
        report["stages"]["instantiate"] = {"ok": True, "seconds": round(time.time() - t0, 2),
                                            "type": type(dl).__name__}
    except Exception as e:
        report["stages"]["instantiate"] = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
        _emit(report, args.out)
        print(f"INSTANTIATE FAIL: {e}", file=sys.stderr)
        return 2

    # Stage 3: pull batches
    expected = {k.strip() for k in args.expected_keys.split(",") if k.strip()}
    batches = []
    t0 = time.time()
    try:
        it = iter(dl)
        for i in range(args.num_batches):
            b = next(it)
            batches.append(_describe(b))
        report["stages"]["iterate"] = {"ok": True, "seconds": round(time.time() - t0, 2),
                                        "n_batches": len(batches)}
    except StopIteration:
        report["stages"]["iterate"] = {"ok": False, "error": "StopIteration on first batch (empty dataset?)"}
        _emit(report, args.out)
        return 3
    except Exception as e:
        report["stages"]["iterate"] = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
        _emit(report, args.out)
        print(f"ITERATE FAIL: {e}", file=sys.stderr)
        return 3

    report["sample_batches"] = batches

    # Stage 4: verify
    problems = []
    for i, b in enumerate(batches):
        if expected and isinstance(b, dict):
            missing = expected - set(b.keys())
            if missing:
                problems.append(f"batch {i}: missing keys {sorted(missing)}")
        # walk for finite=False
        def walk(x, path):
            if isinstance(x, dict):
                if "finite" in x and x["finite"] is False:
                    problems.append(f"batch {i} {path}: NaN/Inf detected")
                for k, v in x.items():
                    if k not in {"shape", "dtype", "finite", "device", "type", "repr"}:
                        walk(v, f"{path}.{k}")
            elif isinstance(x, list):
                for j, v in enumerate(x):
                    walk(v, f"{path}[{j}]")
        walk(b, "")
    report["stages"]["verify"] = {"ok": not problems, "problems": problems}

    _emit(report, args.out)

    print(json.dumps({"summary": {k: v.get("ok") for k, v in report["stages"].items()}}, indent=2))
    if problems:
        print(f"VERIFY FAIL: {problems}", file=sys.stderr)
        return 4
    return 0


def _emit(report: dict, out: str) -> None:
    if out:
        Path(out).write_text(json.dumps(report, indent=2, default=str))
    else:
        print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
