---
name: test_policy_connection
description: Test connection to a running policy WebSocket server and run a sample inference
version: 2.0.0
category: test
parameters:
  host:
    type: string
    description: "Hostname of the policy server (e.g. cn06 or localhost)"
    default: "localhost"
  port:
    type: integer
    description: "Port of the policy server"
    default: 18800
requires:
  bins: [python3]
timeout: 30
command_template: |
  PYTHONPATH=$AGENTROBOT_ROOT/agentic/policy_websocket/src:${PYTHONPATH:-}
  export PYTHONPATH
  python3 -c "
  from policy_websocket import WebsocketClientPolicy
  import numpy as np
  import time

  print('Connecting to {host}:{port}...')
  try:
      policy = WebsocketClientPolicy(host='{host}', port={port})
  except Exception as e:
      print(f'Connection failed: {e}')
      exit(1)

  metadata = policy.get_server_metadata()
  print(f'Connected! Metadata: {metadata}')

  # Init
  init_obs = {'action_dim': 7, 'task_description': 'test'}
  r = policy.infer(init_obs)
  print(f'Init OK: actions shape={r[\"actions\"].shape}')

  # Inference with fake images
  obs = {
      'agentview_image': np.random.randint(0,255,(256,256,3),dtype=np.uint8),
      'robot0_eye_in_hand_image': np.random.randint(0,255,(256,256,3),dtype=np.uint8),
      'robot0_eef_pos': np.zeros(3),
      'robot0_eef_quat': np.array([0,0,0,1.0]),
      'robot0_joint_pos': np.zeros(7),
      'robot0_gripper_qpos': np.array([0.04,0.04]),
      'task_description': 'pick up the red cube',
  }
  t0 = time.time()
  r = policy.infer(obs)
  dt = time.time() - t0
  print(f'Inference OK: actions={r[\"actions\"].flatten()[:4]}..., time={dt*1000:.0f}ms')
  print(f'Server timing: {r.get(\"server_timing\", {})}')
  policy.close()
  print('All tests passed.')
  "
---

# Test Policy Connection

Connects to a running policy server, sends a test observation, and verifies inference works.
Use this after deploying a policy to confirm it's ready. Defaults to localhost.
