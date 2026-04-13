---
name: list_files
description: "List files matching a glob pattern under a directory (auto-skips .venv, __pycache__, .git, data/, checkpoints/). Use this to discover demo scripts, model definitions, and config files inside an arbitrary repo before reading them."
version: 1.0.0
category: util
parameters:
  root:
    type: string
    description: "Absolute path to the directory to list"
    required: true
  pattern:
    type: string
    description: "Glob pattern (e.g. '*.py', 'demo*.py', 'config*.yaml'). Default: '*.py'"
    default: "*.py"
  max_depth:
    type: integer
    description: "Maximum recursion depth (default 6)"
    default: 6
requires:
  bins: [python3]
timeout: 30
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/list_files/list.py "{root}" --pattern "{pattern}" --max-depth {max_depth}
---

# List Files

Walks a directory and prints files matching a glob, skipping common noise dirs
(`.venv`, `__pycache__`, `.git`, `data`, `checkpoints`, `logs`, `wandb`, ...).
Output is capped at 500 entries.

## When to use
- After cloning a new repo: `list_files(root="...", pattern="demo*.py")` to find
  example inference scripts.
- To locate config files: `pattern="*.yaml"` or `pattern="config*.py"`.
- To find a model class definition: `pattern="*model*.py"`.

## Tips
- Pair with `read_file`: list first, then read the most promising hits.
- If output is truncated, narrow the pattern or pick a subdirectory.
