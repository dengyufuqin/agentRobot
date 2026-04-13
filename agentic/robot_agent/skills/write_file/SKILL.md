---
name: write_file
description: "Write content to a file via base64 encoding. Use this to create policy_server.py adapters, config files, or scripts with arbitrary content."
version: 1.0.0
category: util
parameters:
  file_path:
    type: string
    description: "Absolute path to the file to write"
    required: true
  content_b64:
    type: string
    description: "Base64-encoded file content (use base64 encoding to avoid shell escaping issues)"
    required: true
  make_executable:
    type: string
    description: "If 'true', make the file executable (chmod +x)"
    default: "false"
requires:
  bins: [python3]
timeout: 10
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/write_file/write.py "{file_path}" --b64 "{content_b64}"
---

# Write File

Utility skill that writes content to a file using base64 encoding (safe for any content).
Used by the agent to generate policy_server.py adapters with real model loading code.
