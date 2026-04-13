---
name: create_deploy_skill
description: "Auto-generate a new deploy SKILL.md for a wrapped repository. Creates a complete skill directory with deployment capabilities — the OpenClaw skill-creator pattern."
version: 1.1.0
category: meta
parameters:
  repo_path:
    type: string
    description: "Absolute path to the repository (must have policy_server.py)"
    required: true
  skill_name:
    type: string
    description: "Name for the new skill (e.g. deploy_octo, deploy_rt2)"
    required: true
  default_checkpoint:
    type: string
    description: "Default model checkpoint path"
    default: ""
  gpu_memory_gb:
    type: integer
    description: "Estimated GPU memory needed in GB"
    default: 16
  load_time_seconds:
    type: integer
    description: "Estimated model load time in seconds"
    default: 90
  action_dim:
    type: integer
    description: "Output action dimension"
    default: 7
  env_setup:
    type: string
    description: "Environment setup commands (e.g. 'source .venv/bin/activate')"
    default: "source .venv/bin/activate"
requires:
  bins: [bash, mkdir]
timeout: 15
command_template: |
  bash $AGENTROBOT_ROOT/agentic/robot_agent/skills/create_deploy_skill/generate.sh "{repo_path}" "{skill_name}" "{default_checkpoint}" "{gpu_memory_gb}" "{load_time_seconds}" "{action_dim}" "{env_setup}"
---

# Create Deploy Skill

Meta-skill that auto-generates a new deployment SKILL.md for a wrapped repository.
This is the **skill-creator** pattern from OpenClaw — a skill that creates other skills.

## What it generates
A complete deploy skill with:
- SSH-based deployment to compute nodes
- Port conflict checking
- Environment setup (venv/apptainer)
- Logging to shared filesystem
- GPU device selection

## Flow
```
analyze_repo → wrap_policy → create_deploy_skill → [new skill ready]
```
