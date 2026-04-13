---
name: read_file
description: "Read a source file (or a line range) with line numbers. Use this to inspect any file in a cloned repo BEFORE generating a policy_server.py adapter — never write adapter code without first reading the model's actual loading/inference scripts."
version: 1.0.0
category: util
parameters:
  file_path:
    type: string
    description: "Absolute path to the file to read"
    required: true
  start:
    type: integer
    description: "1-indexed start line (optional, default 1)"
    default: 1
  end:
    type: integer
    description: "1-indexed end line inclusive (optional, default = end of file or +2000 lines)"
    default: 0
requires:
  bins: [python3]
timeout: 15
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/read_file/read.py "{file_path}" --start {start} $( [ "{end}" != "0" ] && echo --end {end} )
---

# Read File

Inspect a file with line numbers. Output is capped at 2000 lines / 200KB per call.
For larger files, paginate by passing `start` and `end`.

## When to use
- BEFORE calling `wrap_policy`: read the repo's demo/inference scripts to understand
  how the model is loaded and how `predict()` is called.
- BEFORE calling `fix_deps`: read the failing import line in pyproject.toml /
  requirements.txt to see what's actually being asked for.
- AFTER `validate_policy_server` fails: read the traceback to know what to fix.
