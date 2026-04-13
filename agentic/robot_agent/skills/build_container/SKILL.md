---
name: build_container
description: "Build an Apptainer container for a repository with complex dependencies (JAX+torch, TF, etc). Creates a .sif file for isolated deployment."
version: 1.0.0
category: env
parameters:
  repo_path:
    type: string
    description: "Absolute path to the repository"
    required: true
  framework:
    type: string
    description: "Primary ML framework: torch, jax, or jax-torch (for mixed)"
    default: "torch"
  python_version:
    type: string
    description: "Python version"
    default: "3.11"
  extra_deps:
    type: string
    description: "Additional pip packages (space-separated)"
    required: false
requires:
  bins: [apptainer]
timeout: 900
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/build_container/generate_def.py \
    "{repo_path}" --framework "{framework}" --extra-deps "{extra_deps}" --build
---

# Build Container

Creates an Apptainer container (.sif) for repos with complex dependencies.
Use this instead of setup_env when:
- JAX + PyTorch mixed dependencies
- TensorFlow + JAX conflicts
- System library requirements (CUDA toolkit, ffmpeg, etc)

The container:
- Uses NVIDIA NGC base images (pre-built CUDA + framework)
- Installs repo dependencies inside the container
- Mounts VAST filesystem for checkpoints and logs
- Runs policy_server.py as default entrypoint
