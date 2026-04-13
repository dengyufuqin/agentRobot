#!/usr/bin/env python3
"""
Diagnose and auto-fix dependency issues in a Python venv.

Runs import tests, pattern-matches errors against known fixes, applies them,
and retries until all imports pass or max retries exhausted.

Usage:
    python fix_deps.py /path/to/repo [--modules torch,numpy,robomimic] [--max-retries 5]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Known fix patterns: (regex on stderr/stdout) → list of fix commands
# Each fix is a dict with: pattern, description, commands (list of shell cmds)
# Commands use {venv} placeholder for the venv python/pip path
# ---------------------------------------------------------------------------

FIX_PATTERNS = [
    # --- Missing packages (most common) ---
    {
        "pattern": r"ModuleNotFoundError: No module named '(\w+)'",
        "description": "Missing package (auto-detect)",
        "handler": "missing_module",
    },
    # --- libGL / OpenCV headless ---
    {
        "pattern": r"ImportError:.*libGL\.so",
        "description": "OpenCV needs headless variant (no libGL on HPC)",
        "commands": [
            "uv pip install --python {python} opencv-python-headless",
            "uv pip uninstall --python {python} opencv-python 2>/dev/null; true",
        ],
    },
    # --- numpy 2.x incompatibility ---
    {
        "pattern": r"(A module that was compiled using NumPy 1\.x|numpy\.core\._multiarray_umath|numpy\.dtype size changed|module 'numpy' has no attribute)",
        "description": "numpy 2.x incompatible with compiled extensions",
        "commands": [
            'uv pip install --python {python} "numpy<2"',
        ],
    },
    # --- mujoco / robosuite version mismatch ---
    {
        "pattern": r"(mujoco.*SIGABRT|mujoco.*core dump|robosuite.*mujoco|mj_loadXML|mjModel)",
        "description": "mujoco version incompatible with robosuite 1.4.x",
        "commands": [
            'uv pip install --python {python} "mujoco==2.3.7"',
        ],
    },
    # --- torch.xpu / diffusers too new ---
    {
        "pattern": r"AttributeError: module 'torch' has no attribute 'xpu'",
        "description": "diffusers version too new for installed torch",
        "commands": [
            'uv pip install --python {python} "diffusers<0.27"',
        ],
    },
    # --- cmake required for building extensions ---
    {
        "pattern": r"(RuntimeError: CMake must be installed|No CMAKE_CUDA_COMPILER could be found)",
        "description": "cmake needed for native extensions",
        "commands": [
            "uv pip install --python {python} cmake",
        ],
    },
    # --- EGL / display issues (informational, not fixable) ---
    {
        "pattern": r"eglQueryString|EGL_EXT_device_base|DISPLAY.*not set",
        "description": "EGL/display error — needs GPU node, not fixable on login node",
        "commands": [],  # empty = skip, just report
        "severity": "info",
    },
    # --- scipy missing (common transitive dep) ---
    {
        "pattern": r"ModuleNotFoundError: No module named 'scipy'",
        "description": "Missing scipy",
        "commands": [
            'uv pip install --python {python} scipy "numpy<2"',
        ],
    },
    # --- PIL / Pillow ---
    {
        "pattern": r"ModuleNotFoundError: No module named 'PIL'",
        "description": "Missing Pillow",
        "commands": [
            "uv pip install --python {python} Pillow",
        ],
    },
    # --- torch.load weights_only default changed in 2.6+ ---
    {
        "pattern": r"_pickle\.UnpicklingError.*Weights only load failed",
        "description": "PyTorch 2.6+ changed torch.load default to weights_only=True",
        "commands": [],  # Needs code patch, not pip fix
        "severity": "warn",
        "suggestion": "Patch the torch.load() call to add weights_only=False",
    },
    # --- CUDA version mismatch ---
    {
        "pattern": r"(CUDA error.*no kernel image|CUDA.*not compatible.*compute capability)",
        "description": "CUDA compute capability mismatch",
        "commands": [
            'uv pip install --python {python} "torch>=2.2,<2.8" "torchvision" '
            "--index-url https://download.pytorch.org/whl/cu121 "
            "--extra-index-url https://pypi.org/simple/ "
            "--index-strategy unsafe-best-match",
        ],
    },
]

# Common missing modules → pip package name mapping
MODULE_TO_PACKAGE = {
    "cv2": "opencv-python-headless",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "attr": "attrs",
    "bs4": "beautifulsoup4",
    "google.protobuf": "protobuf",
    "ruamel": "ruamel.yaml",
    "usb": "pyusb",
    "serial": "pyserial",
    "wx": "wxPython",
    "gi": "PyGObject",
    "Crypto": "pycryptodome",
    "dateutil": "python-dateutil",
    "dotenv": "python-dotenv",
    "git": "gitpython",
    "IPython": "ipython",
    "jwt": "PyJWT",
    "magic": "python-magic",
    "opengl": "PyOpenGL",
    "zmq": "pyzmq",
    "msgpack": "msgpack",
    "websockets": "websockets",
    "future": "future",
    "wandb": "wandb",
    "tqdm": "tqdm",
    "h5py": "h5py",
    "hydra": "hydra-core",
    "omegaconf": "omegaconf",
    "einops": "einops",
    "dm_control": "dm-control",
    "gym": "gymnasium",
    "mujoco_py": "mujoco-py",
    "robosuite": "robosuite",
    "dill": "dill",
    "psutil": "psutil",
    "imageio": "imageio",
    "trimesh": "trimesh",
    "pybullet": "pybullet",
    "plotly": "plotly",
    "mediapy": "mediapy",
    "flax": "flax",
    "optax": "optax",
    "chex": "chex",
    "distrax": "distrax",
    "matplotlib": "matplotlib",
}


def get_venv_python(repo_path: str) -> str:
    """Get the venv python path for a repo."""
    venv = Path(repo_path) / ".venv" / "bin" / "python3"
    if venv.exists():
        return str(venv)
    raise FileNotFoundError(f"No venv found at {repo_path}/.venv")


def detect_importable_modules(repo_path: str) -> list[str]:
    """Auto-detect modules that should be importable from the repo."""
    repo = Path(repo_path)
    modules = []

    # From pyproject.toml
    pyproject = repo / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text()
        # Look for [project] name = "..."
        m = re.search(r'\[project\]\s*\n(?:.*\n)*?name\s*=\s*["\']([^"\']+)', text)
        if m:
            pkg = m.group(1).replace("-", "_")
            modules.append(pkg)

    # From setup.py / setup.cfg
    setup_py = repo / "setup.py"
    if setup_py.exists():
        text = setup_py.read_text()
        m = re.search(r"name\s*=\s*['\"]([^'\"]+)", text)
        if m:
            pkg = m.group(1).replace("-", "_")
            modules.append(pkg)

    # Always test policy_websocket
    modules.append("policy_websocket")

    # Check for common frameworks based on requirements
    for req_file in ["requirements.txt", "pyproject.toml", "setup.py"]:
        p = repo / req_file
        if p.exists():
            text = p.read_text().lower()
            if "torch" in text:
                modules.append("torch")
            if "jax" in text:
                modules.append("jax")
            if "tensorflow" in text:
                modules.append("tensorflow")
            if "transformers" in text:
                modules.append("transformers")
            if "robomimic" in text:
                modules.append("robomimic")
            if "robosuite" in text:
                modules.append("robosuite")

    return list(dict.fromkeys(modules))  # dedupe preserving order


def test_imports(python: str, modules: list[str], extra_pythonpath: str = "") -> list[dict]:
    """Test importing each module, return list of failures with error details."""
    failures = []
    env = None
    if extra_pythonpath:
        env = {**os.environ, "PYTHONPATH": extra_pythonpath}
    for mod in modules:
        result = subprocess.run(
            [python, "-c", f"import {mod}; print('{mod}: OK')"],
            capture_output=True, text=True, timeout=30, env=env,
        )
        if result.returncode != 0:
            failures.append({
                "module": mod,
                "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip(),
                "error": (result.stderr + result.stdout).strip(),
            })
    return failures


def match_fix(error_text: str) -> list[dict]:
    """Match error text against known fix patterns, return applicable fixes."""
    matches = []
    for fix in FIX_PATTERNS:
        m = re.search(fix["pattern"], error_text, re.IGNORECASE)
        if m:
            matches.append({**fix, "match": m})
    return matches


def handle_missing_module(error_text: str, python: str) -> list[str]:
    """Handle ModuleNotFoundError — figure out the right pip package."""
    m = re.search(r"No module named '([^']+)'", error_text)
    if not m:
        return []
    mod_name = m.group(1).split(".")[0]  # top-level module

    # Check our mapping first
    pkg = MODULE_TO_PACKAGE.get(mod_name, mod_name)

    return [f"uv pip install --python {python} {pkg}"]


def apply_fix(commands: list[str], python: str) -> tuple[bool, str]:
    """Run fix commands, return (success, output)."""
    outputs = []
    for cmd in commands:
        cmd = cmd.replace("{python}", python)
        print(f"  [FIX] {cmd}")
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=120,
        )
        out = result.stdout.strip()
        if result.stderr:
            out += "\n" + result.stderr.strip()
        outputs.append(out)
        if result.returncode != 0:
            return False, "\n".join(outputs)
    return True, "\n".join(outputs)


def pin_numpy_if_needed(python: str):
    """Check if numpy 2.x is installed and pin to <2 preemptively."""
    result = subprocess.run(
        [python, "-c", "import numpy; print(numpy.__version__)"],
        capture_output=True, text=True, timeout=10,
    )
    if result.returncode == 0:
        ver = result.stdout.strip()
        if ver.startswith("2."):
            print(f"  [PREEMPTIVE] numpy {ver} detected, pinning to <2")
            apply_fix(['uv pip install --python {python} "numpy<2"'], python)


def run_diagnosis(repo_path: str, modules: list[str] = None, max_retries: int = 5,
                  extra_pythonpath: str = "") -> dict:
    """Main diagnostic loop: test → fix → retry."""
    python = get_venv_python(repo_path)
    print(f"=== Dependency Diagnosis for {Path(repo_path).name} ===")
    print(f"Python: {python}")

    # Auto-detect PYTHONPATH for nested repos
    if not extra_pythonpath:
        repo = Path(repo_path)
        # Check for nested directories with the same name (common pattern)
        for subdir in repo.iterdir():
            if subdir.is_dir() and (subdir / "setup.py").exists():
                extra_pythonpath = str(subdir)
                print(f"Auto-detected nested repo: {subdir.name}")
                break
    if extra_pythonpath:
        print(f"PYTHONPATH: {extra_pythonpath}")

    if not modules:
        modules = detect_importable_modules(repo_path)
    print(f"Testing modules: {modules}")

    # Preemptive numpy check
    pin_numpy_if_needed(python)

    fixes_applied = []
    warnings = []
    round_num = 0
    failed_installs = set()  # Track packages that failed to resolve

    for round_num in range(1, max_retries + 1):
        print(f"\n--- Round {round_num}/{max_retries} ---")
        failures = test_imports(python, modules, extra_pythonpath)

        if not failures:
            print("\nAll imports passed!")
            break

        print(f"  {len(failures)} failure(s):")
        applied_this_round = False

        for fail in failures:
            print(f"    - {fail['module']}: {fail['error'][:120]}")
            matched = match_fix(fail["error"])
            fixed = False

            if not matched:
                # Try generic missing module handler
                if "ModuleNotFoundError" in fail["error"]:
                    mod_match = re.search(r"No module named '([^']+)'", fail["error"])
                    missing_mod = mod_match.group(1).split(".")[0] if mod_match else None
                    if missing_mod and missing_mod in failed_installs:
                        warnings.append({
                            "module": fail["module"],
                            "error": f"Package '{missing_mod}' cannot be installed via pip (may need compilation or manual install)",
                            "suggestion": f"Try: pip install {missing_mod} --no-build-isolation, or build from source",
                            "severity": "warn",
                        })
                        continue
                    cmds = handle_missing_module(fail["error"], python)
                    if cmds:
                        ok, _ = apply_fix(cmds, python)
                        fixes_applied.append({
                            "module": fail["module"],
                            "description": f"Missing package: {missing_mod or '?'}",
                            "fix": cmds[0],
                            "success": ok,
                        })
                        if not ok and missing_mod:
                            failed_installs.add(missing_mod)
                        applied_this_round = True
                        continue
                # Unknown error
                warnings.append({
                    "module": fail["module"],
                    "error": fail["error"][:200],
                    "suggestion": "Unknown error pattern — manual investigation needed",
                })
                continue

            for fix in matched:
                if fixed:
                    break  # one fix per failure per round
                severity = fix.get("severity", "error")
                if severity == "info":
                    warnings.append({
                        "module": fail["module"],
                        "description": fix["description"],
                        "severity": "info",
                    })
                    continue
                if severity == "warn":
                    warnings.append({
                        "module": fail["module"],
                        "description": fix["description"],
                        "suggestion": fix.get("suggestion", ""),
                        "severity": "warn",
                    })
                    continue

                if fix.get("handler") == "missing_module":
                    mod_match = re.search(r"No module named '([^']+)'", fail["error"])
                    missing_mod = mod_match.group(1).split(".")[0] if mod_match else None
                    if missing_mod and missing_mod in failed_installs:
                        warnings.append({
                            "module": fail["module"],
                            "error": f"Package '{missing_mod}' cannot be installed via pip (may need compilation or manual install)",
                            "suggestion": f"Try building from source or use a different Python version",
                            "severity": "warn",
                        })
                        fixed = True
                        continue
                    cmds = handle_missing_module(fail["error"], python)
                else:
                    cmds = [c.replace("{python}", python) for c in fix.get("commands", [])]

                if cmds:
                    ok, _ = apply_fix(cmds, python)
                    fixes_applied.append({
                        "module": fail["module"],
                        "description": fix["description"],
                        "fix": cmds[0],
                        "success": ok,
                    })
                    if not ok:
                        mod_match = re.search(r"No module named '([^']+)'", fail["error"])
                        if mod_match:
                            failed_installs.add(mod_match.group(1).split(".")[0])
                    applied_this_round = True
                    fixed = True

        if not applied_this_round:
            print("  No fixable errors found, stopping.")
            break

        # Track modules that still fail after fix was "successful"
        # (e.g., pip reports success but import still fails — wrong package or needs compilation)
        still_failing = test_imports(python, [f["module"] for f in failures], extra_pythonpath)
        for sf in still_failing:
            mod_match = re.search(r"No module named '([^']+)'", sf["error"])
            if mod_match:
                pkg = mod_match.group(1).split(".")[0]
                if pkg not in failed_installs:
                    failed_installs.add(pkg)
                    # First time seeing repeated failure — allow one more try
                else:
                    pass  # Already tracked, will be caught in next round

        # Re-check numpy after any install (transitive deps can upgrade it)
        pin_numpy_if_needed(python)

    # Final check
    final_failures = test_imports(python, modules, extra_pythonpath)

    result = {
        "repo": str(repo_path),
        "rounds": round_num,
        "fixes_applied": fixes_applied,
        "warnings": warnings,
        "final_failures": [f["module"] for f in final_failures],
        "all_passed": len(final_failures) == 0,
    }

    # Summary
    print(f"\n=== Diagnosis Complete ===")
    print(f"Rounds: {round_num}")
    print(f"Fixes applied: {len(fixes_applied)}")
    for f in fixes_applied:
        status = "OK" if f["success"] else "FAILED"
        print(f"  [{status}] {f.get('description', f['fix'])}")
    if warnings:
        print(f"Warnings: {len(warnings)}")
        for w in warnings:
            print(f"  [{w.get('severity', 'warn')}] {w.get('description', w.get('error', '?'))}: {w.get('suggestion', '')}")
    if final_failures:
        print(f"Still failing: {[f['module'] for f in final_failures]}")
    else:
        print("All imports passing!")

    return result


def main():
    parser = argparse.ArgumentParser(description="Diagnose and fix Python dependency issues")
    parser.add_argument("repo_path", help="Path to the repository")
    parser.add_argument("--modules", default=None, help="Comma-separated list of modules to test")
    parser.add_argument("--max-retries", type=int, default=5, help="Max fix-retry rounds")
    parser.add_argument("--pythonpath", default="", help="Extra PYTHONPATH for nested repos")
    parser.add_argument("--json", action="store_true", help="Output result as JSON")
    args = parser.parse_args()

    modules = args.modules.split(",") if args.modules else None
    result = run_diagnosis(args.repo_path, modules=modules, max_retries=args.max_retries,
                          extra_pythonpath=args.pythonpath)

    if args.json:
        print("\n" + json.dumps(result, indent=2))

    sys.exit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
