#!/usr/bin/env python3
"""Generate an Apptainer .def file for a repo and optionally build it."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

_AGENT_ROOT = os.environ.get("AGENTROBOT_ROOT", str(Path(__file__).resolve().parent.parent.parent.parent.parent))
CONTAINER_DIR = Path(_AGENT_ROOT) / "containers"
POLICY_WS = str(Path(_AGENT_ROOT) / "agentic" / "policy_websocket")

BASE_IMAGES = {
    "torch": "nvcr.io/nvidia/pytorch:24.01-py3",
    "jax": "nvcr.io/nvidia/jax:24.04-py3",
    "jax-torch": "nvcr.io/nvidia/jax:24.04-py3",
}


def generate_def(repo_path: str, framework: str, extra_deps: str) -> str:
    repo_name = os.path.basename(repo_path)
    base_image = BASE_IMAGES.get(framework, BASE_IMAGES["torch"])

    extra_install = ""
    if extra_deps.strip():
        extra_install = f"    pip install --no-cache-dir {extra_deps}\n"

    return f"""Bootstrap: docker
From: {base_image}

%files
    {repo_path} /opt/{repo_name}
    {POLICY_WS} /opt/policy_websocket

%post
    pip install --no-cache-dir websockets msgpack
    pip install --no-cache-dir -e /opt/policy_websocket
    cd /opt/{repo_name}
    if [ -f pyproject.toml ]; then
        pip install --no-cache-dir -e "." || pip install --no-cache-dir -e "." --no-deps
    elif [ -f requirements.txt ]; then
        pip install --no-cache-dir -r requirements.txt
    elif [ -f setup.py ]; then
        pip install --no-cache-dir -e .
    fi
{extra_install}
%environment
    export PYTHONPATH=/opt/policy_websocket/src:/opt/{repo_name}

%runscript
    cd /opt/{repo_name}
    exec python3 policy_server.py "$@"

%test
    python3 -c "from policy_websocket import BasePolicy; print('policy_websocket OK')"

%labels
    Author RobotOpsAgent
    Framework {framework}
    Repo {repo_name}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_path")
    parser.add_argument("--framework", default="torch")
    parser.add_argument("--extra-deps", default="")
    parser.add_argument("--build", action="store_true", help="Also build the .sif")
    args = parser.parse_args()

    repo_name = os.path.basename(args.repo_path)
    CONTAINER_DIR.mkdir(parents=True, exist_ok=True)
    def_path = CONTAINER_DIR / f"{repo_name}.def"
    sif_path = CONTAINER_DIR / f"{repo_name}.sif"

    content = generate_def(args.repo_path, args.framework, args.extra_deps)
    def_path.write_text(content)
    print(f"Definition file: {def_path}")

    if args.build:
        print(f"\nBuilding {sif_path} (this may take 5-10 minutes)...")
        result = subprocess.run(
            ["apptainer", "build", "--fakeroot", str(sif_path), str(def_path)],
            capture_output=False,
        )
        if sif_path.exists():
            size = sif_path.stat().st_size / (1024 * 1024)
            print(f"\n=== Container built: {sif_path} ({size:.0f}MB) ===")
            print(f"Run: apptainer run --nv {sif_path} --port 18800")
        else:
            print("ERROR: Build failed")
            sys.exit(1)
    else:
        print(f"\nTo build: apptainer build --fakeroot {sif_path} {def_path}")


if __name__ == "__main__":
    main()
