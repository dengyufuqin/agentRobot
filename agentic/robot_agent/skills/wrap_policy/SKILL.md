---
name: wrap_policy
description: "[DEPRECATED — kept for back-compat only] Regex-based adapter generator for 6 common patterns (Hydra, from_pretrained, algo_factory, etc.). For any real adapter work, prefer probe_run → extract_io_spec → write_file → validate_policy_server — that flow works on ANY repo, not just the common-pattern subset."
version: 1.1.0
category: meta
parameters:
  repo_path:
    type: string
    description: "Absolute path to the cloned repository (from analyze_repo output)"
    required: true
  model_class:
    type: string
    description: "Python class name for the model (e.g. OctoModel, OpenVLAPolicy)"
    required: true
  model_module:
    type: string
    description: "Python module path to import model from (e.g. octo.model.octo_model)"
    required: true
  checkpoint:
    type: string
    description: "Default checkpoint path (HuggingFace ID or local path)"
    default: ""
  action_dim:
    type: integer
    description: "Output action dimension"
    default: 7
  framework:
    type: string
    description: "ML framework: torch, jax, or tensorflow"
    default: "torch"
requires:
  bins: [bash, cp, sed]
timeout: 30
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/wrap_policy/generate_smart.py "{repo_path}" "{model_class}" "{model_module}" "{checkpoint}" "{action_dim}" "{framework}"
---

# Wrap Policy

Meta-skill that generates a `policy_server.py` adapter for any repository.

## How it works
1. Takes repo path and model info from `analyze_repo` output
2. Copies a template implementing `BasePolicy` interface
3. Fills in model-specific placeholders (class name, module, checkpoint)
4. Generates a config file for further customization

## After generation
The adapter needs manual editing to:
- Import the actual model class
- Implement model loading in `__init__`
- Map observations from client format to model format
- Post-process model output to standard action format
