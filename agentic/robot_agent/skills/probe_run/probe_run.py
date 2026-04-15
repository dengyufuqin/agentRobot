#!/usr/bin/env python3
"""Probe-run a repo's own entry script until it reaches a success marker, then kill.

Purpose:
  1. Verify the repo actually works with the configured checkpoint/env.
  2. Optionally capture first-call tensor shapes (--io-spec-hook).
  3. Fail fast on obvious errors (traceback, OOM, import error).

Exit codes:
  0  success marker seen, process killed cleanly (expected happy path)
  1  bad args / entry not found
  2  timeout without marker
  3  process exited with code 0 but no marker seen (probably exits too early)
  4  error pattern detected in output (traceback / OOM / import error)
  >4 process exited with non-zero code on its own (propagated)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_SUCCESS_MARKERS = [
    r"\bstep[:\s]+(1|10|100)\b",            # training step reached
    r"[Ee]poch[:\s]+0",                      # first epoch
    r"[Ss]erver.*(?:listen|ready|start)",    # server up
    r"Uvicorn running",
    r"WebSocket.*(?:ready|listen)",
    r"[Mm]odel.*loaded",
    r"[Ll]oaded policy|Policy ready",
    r"[Ii]nference.*ready",
    r"Started policy server",
    # NOTE: do NOT add "Warming up model" here — it fires BEFORE the first forward
    # call. For shape-hook use-cases, the [PROBE-HOOK] marker (auto-added when
    # io_spec_hook=True) is the correct post-forward trigger.
]
DEFAULT_ERROR_MARKERS = [
    r"Traceback \(most recent call last\)",
    r"CUDA out of memory",
    r"ImportError",
    r"ModuleNotFoundError",
    r"AssertionError",
    r"RuntimeError:.*(?:CUDA|shape|size mismatch)",
]


def build_hook_preamble(hook_out: Path, target_classes: list[str]) -> str:
    """Return a Python snippet that installs TWO hooks:
      1. An `nn.Module.__init__` patch that detects when a target policy class
         is instantiated, then monkey-patches THAT CLASS's `.forward` method.
         This handles the common pi0/vla idiom where code calls `.forward()`
         directly instead of `module(x)` — bypassing normal __call__ hooks.
      2. A fallback `nn.Module.__call__` patch that captures ANY top-level
         call at depth 0. Used when target_classes is empty or hasn't matched.

    Writes-through to JSON on every capture (SIGTERM-safe; atexit won't run).
    """
    return f'''
import os as _os, json as _json
_HOOK_OUT = {str(hook_out)!r}
_TARGETS = {target_classes!r}
_HOOK_STATE = {{"captured": False, "calls": [], "patched_classes": set()}}

def _shape(x):
    if hasattr(x, "shape"):
        return {{"shape": list(x.shape), "dtype": str(getattr(x, "dtype", None))}}
    if isinstance(x, (list, tuple)):
        return [_shape(v) for v in x]
    if isinstance(x, dict):
        return {{k: _shape(v) for k, v in x.items()}}
    return str(type(x).__name__)

def _dump_now():
    try:
        with open(_HOOK_OUT + ".tmp", "w") as _f:
            _json.dump(_HOOK_STATE["calls"], _f, indent=2, default=str)
        _os.replace(_HOOK_OUT + ".tmp", _HOOK_OUT)
    except Exception:
        pass

def _record(module, args, kwargs, out, tag):
    try:
        info = {{
            "module_class": type(module).__name__,
            "source": tag,
            "input_args": [_shape(x) for x in args],
            "input_kwargs": {{k: _shape(v) for k, v in kwargs.items()}},
            "output": _shape(out),
        }}
        _HOOK_STATE["calls"].append(info)
        _dump_now()
        print(f"[PROBE-HOOK] captured call #{{len(_HOOK_STATE['calls'])}} from {{type(module).__name__}} via {{tag}}", flush=True)
    except Exception as _e:
        print(f"[PROBE-HOOK] record failed: {{_e}}", flush=True)

def _class_matches(cls):
    if not _TARGETS:
        return False
    name = cls.__name__
    return any(pat in name for pat in _TARGETS)

def _patch_class_forward(cls):
    if cls in _HOOK_STATE["patched_classes"]:
        return
    orig = cls.forward
    def _patched_forward(self, *a, **kw):
        out = orig(self, *a, **kw)
        if not _HOOK_STATE["captured"]:
            _record(self, a, kw, out, tag="class_forward")
            if len(_HOOK_STATE["calls"]) >= 3:
                _HOOK_STATE["captured"] = True
        return out
    cls.forward = _patched_forward
    _HOOK_STATE["patched_classes"].add(cls)
    print(f"[PROBE-HOOK] patched forward on target class {{cls.__name__}}", flush=True)

def _install():
    try:
        import torch.nn as _nn
    except ImportError:
        return

    # --- 1. __init__ hook: patch forward on target classes at instantiation ---
    if _TARGETS:
        _orig_init = _nn.Module.__init__
        def _patched_init(self, *a, **kw):
            _orig_init(self, *a, **kw)
            cls = type(self)
            if _class_matches(cls):
                _patch_class_forward(cls)
        _nn.Module.__init__ = _patched_init

    # --- 2. __call__ fallback: top-level capture ---
    _orig_call = _nn.Module.__call__
    _depth = {{"n": 0}}
    def _patched_call(self, *a, **kw):
        _depth["n"] += 1
        try:
            out = _orig_call(self, *a, **kw)
        finally:
            _depth["n"] -= 1
        if _depth["n"] == 0 and not _HOOK_STATE["captured"]:
            _record(self, a, kw, out, tag="__call__")
            if len(_HOOK_STATE["calls"]) >= 5:
                _HOOK_STATE["captured"] = True
                _nn.Module.__call__ = _orig_call
        return out
    _nn.Module.__call__ = _patched_call

_install()
'''


def run_probe(
    repo_path: Path,
    entry_script: str,
    venv_python: Path,
    timeout: int,
    success_markers: list[str],
    error_markers: list[str],
    extra_args: list[str],
    io_spec_hook: bool,
    target_classes: list[str],
) -> int:
    entry_abs = (repo_path / entry_script).resolve()
    if not entry_abs.is_file():
        print(f"ERROR: entry script not found: {entry_abs}")
        return 1
    if not venv_python.is_file() or not os.access(venv_python, os.X_OK):
        print(f"ERROR: venv python not executable: {venv_python}")
        return 1

    # When io_spec_hook is on, prioritize terminating after capture so we
    # always get the forward-call shapes (many servers print "loaded" BEFORE
    # the first forward — warmup JIT can take minutes, exceeding grace).
    effective_markers = list(success_markers)
    if io_spec_hook:
        effective_markers.insert(0, r"\[PROBE-HOOK\] captured")

    success_re = re.compile("|".join(f"(?:{p})" for p in effective_markers))
    error_re = re.compile("|".join(f"(?:{p})" for p in error_markers))

    # Build the command. If hook requested, prepend a -c preamble before target.
    if io_spec_hook:
        hook_out = repo_path / ".probe_io_spec.json"
        # Write a wrapper script that installs the hook then execfile's the target
        wrapper = repo_path / ".probe_wrapper.py"
        wrapper_content = (
            build_hook_preamble(hook_out, target_classes)
            + f"\n_target = {str(entry_abs)!r}\n"
            + "import sys as _sys\n"
            + f"_sys.argv = [_target] + {extra_args!r}\n"
            + "with open(_target) as _f: _code = _f.read()\n"
            + "exec(compile(_code, _target, 'exec'), {'__name__': '__main__', '__file__': _target})\n"
        )
        wrapper.write_text(wrapper_content)
        cmd = [str(venv_python), "-u", str(wrapper)]
        print(f"[PROBE] IO spec hook enabled -> {hook_out}")
    else:
        cmd = [str(venv_python), "-u", str(entry_abs), *extra_args]

    print(f"[PROBE] cwd: {repo_path}")
    print(f"[PROBE] cmd: {' '.join(cmd)}")
    print(f"[PROBE] timeout: {timeout}s")
    print(f"[PROBE] success markers: {success_markers}")
    print("-" * 60, flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    start = time.time()
    marker_hit: str | None = None
    line_count = 0

    try:
        while True:
            if time.time() - start > timeout:
                elapsed = time.time() - start
                print(f"\n[PROBE] TIMEOUT after {elapsed:.1f}s — no success marker seen", flush=True)
                _terminate(proc)
                return 2

            line = proc.stdout.readline()
            if not line:
                rc = proc.wait()
                elapsed = time.time() - start
                print(f"\n[PROBE] process exited on its own (rc={rc}) after {elapsed:.1f}s", flush=True)
                # NOTE: marker_hit is unreachable here — hitting a marker returns immediately below.
                if rc == 0:
                    print("[PROBE] WARN: clean exit but no marker — script may exit too early")
                    return 3
                return max(5, rc)

            line_count += 1
            sys.stdout.write(line)
            sys.stdout.flush()

            if error_re.search(line):
                # Drain a bit of traceback for context
                for _ in range(25):
                    more = proc.stdout.readline()
                    if not more:
                        break
                    sys.stdout.write(more)
                sys.stdout.flush()
                print(f"\n[PROBE] ERROR pattern detected at line {line_count}", flush=True)
                _terminate(proc)
                return 4

            m = success_re.search(line)
            if m and not marker_hit:
                marker_hit = m.group(0)
                elapsed = time.time() - start
                print(
                    f"\n[PROBE] SUCCESS marker '{marker_hit}' hit at line {line_count} "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
                # let it breathe briefly to flush shape-hook output
                time.sleep(1.5)
                _terminate(proc)
                hook_file = repo_path / ".probe_io_spec.json"
                if io_spec_hook and hook_file.is_file():
                    try:
                        data = json.loads(hook_file.read_text())
                        print(f"[PROBE] IO spec captured ({len(data)} calls) -> {hook_file}")
                    except Exception as e:
                        print(f"[PROBE] IO spec file exists but unreadable: {e}")
                return 0
    except KeyboardInterrupt:
        _terminate(proc)
        return 130


def _terminate(proc: subprocess.Popen) -> None:
    try:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)
    except ProcessLookupError:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", required=True, type=Path)
    ap.add_argument("--entry-script", required=True, help="path relative to repo_path")
    ap.add_argument("--venv-python", type=Path, default=None)
    ap.add_argument("--timeout", type=int, default=500)
    ap.add_argument("--success-markers", default="", help="pipe-separated regex patterns; empty = defaults")
    ap.add_argument("--error-markers", default="", help="pipe-separated regex patterns; empty = defaults")
    ap.add_argument("--extra-args", default="", help="space-separated CLI args for the entry script")
    ap.add_argument("--io-spec-hook", default="false", help="'true' to capture first-call tensor shapes")
    ap.add_argument("--target-classes", default="",
                    help="Comma-separated class-name substrings whose .forward() should be patched "
                         "(e.g. 'PI0Pytorch,OpenVLAForAction'). Needed when the policy calls "
                         "self.forward() directly instead of self(x).")
    args = ap.parse_args()

    repo = args.repo_path.resolve()
    if not repo.is_dir():
        print(f"ERROR: repo_path not a directory: {repo}")
        return 1

    venv_py = args.venv_python or (repo / ".venv" / "bin" / "python3")

    success = (
        [p for p in args.success_markers.split("|") if p] if args.success_markers else DEFAULT_SUCCESS_MARKERS
    )
    error = (
        [p for p in args.error_markers.split("|") if p] if args.error_markers else DEFAULT_ERROR_MARKERS
    )
    extra = args.extra_args.split() if args.extra_args.strip() else []
    targets = [t.strip() for t in args.target_classes.split(",") if t.strip()]

    return run_probe(
        repo_path=repo,
        entry_script=args.entry_script,
        venv_python=venv_py,
        timeout=args.timeout,
        success_markers=success,
        error_markers=error,
        extra_args=extra,
        io_spec_hook=args.io_spec_hook.lower() == "true",
        target_classes=targets,
    )


if __name__ == "__main__":
    sys.exit(main())
