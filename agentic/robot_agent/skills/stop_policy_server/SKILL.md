---
name: stop_policy_server
description: Stop a running policy WebSocket server (local or remote)
version: 2.0.0
category: deploy
parameters:
  node:
    type: string
    description: "Compute node where the server is running. Leave empty for local."
    required: false
  port:
    type: integer
    description: "Port of the server to stop"
    default: 18800
requires:
  bins: [bash]
timeout: 15
command_template: |
  if [ -z "{node}" ] || [ "{node}" = "localhost" ]; then
    # Local mode
    PID=$(ss -tlnp 2>/dev/null | grep :{port} | grep -oP 'pid=\K[0-9]+' | head -1)
    if [ -n "$PID" ]; then
      kill $PID 2>/dev/null && echo "Stopped server PID=$PID on localhost:{port}"
    else
      echo "No server found on localhost:{port}"
    fi
  else
    # Remote mode via SSH
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {node} "
      PID=\$(ss -tlnp | grep :{port} | grep -oP 'pid=\K[0-9]+' | head -1)
      if [ -n \"\$PID\" ]; then
        kill \$PID 2>/dev/null && echo 'Stopped server PID='\$PID' on {node}:{port}'
      else
        echo 'No server found on {node}:{port}'
      fi
    " 2>&1
  fi
---

# Stop Policy Server

Stops a policy WebSocket server by finding and killing the process on the specified port.
- **Local mode**: Leave `node` empty
- **Remote mode**: Specify `node` for SSH-based stop
