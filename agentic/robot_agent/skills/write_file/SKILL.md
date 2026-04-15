---
name: write_file
description: "Write content to a file. Use this to create policy_server.py adapters, config files, or scripts. PREFERRED: pass plaintext via `content` (agent auto-encodes). Use `content_b64` only if you already have base64 bytes."
version: 1.0.0
category: util
parameters:
  file_path:
    type: string
    description: "Absolute path to the file to write"
    required: true
  content:
    type: string
    description: "PLAINTEXT file content. USE THIS — the agent handles base64 encoding for you. Do NOT try to generate base64 yourself."
    required: false
  content_b64:
    type: string
    description: "Base64-encoded file content. Only use if you already have valid b64 bytes. For Python code etc., use `content` instead."
    required: false
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
