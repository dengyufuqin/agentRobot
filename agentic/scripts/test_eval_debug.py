#!/usr/bin/env python3
"""Debug version of eval - find where the crash happens."""
import os
import sys
import signal
import faulthandler

# Enable faulthandler to get traceback on segfault
faulthandler.enable()
os.environ.setdefault("MUJOCO_GL", "egl")

print("Step 1: Importing...", flush=True)
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.run_utils import ARM_CONTROLLER_MAP, enable_joint_pos_observable, pad_action_for_env
from policy_websocket import WebsocketClientPolicy
import numpy as np

print("Step 2: Loading benchmark...", flush=True)
bd = benchmark.get_benchmark_dict()
suite = bd["libero_spatial"]()
task = suite.get_task(0)
init_states = suite.get_task_init_states(0)
print(f"Task: {task.language}", flush=True)

print("Step 3: Connecting to server...", flush=True)
host = sys.argv[1] if len(sys.argv) > 1 else "localhost"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 18800
policy = WebsocketClientPolicy(host=host, port=port)
metadata = policy.get_server_metadata()
print(f"Server metadata: {metadata}", flush=True)

print("Step 4: Sending init obs...", flush=True)
init_obs = {"action_dim": 7, "task_description": task.language}
r = policy.infer(init_obs)
print(f"Init response: actions shape={r['actions'].shape}", flush=True)

print("Step 5: Creating env...", flush=True)
bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
env = OffScreenRenderEnv(bddl_file_name=bddl_file, controller="OSC_POSE",
                         camera_heights=256, camera_widths=256)
env.seed(0)
enable_joint_pos_observable(env)
print("Env created!", flush=True)

print("Step 6: Resetting env...", flush=True)
env.reset()
obs = env.set_init_state(init_states[0])
print(f"Obs keys: {list(obs.keys())[:5]}...", flush=True)

print("Step 7: Running 10 dummy steps...", flush=True)
dummy = [0, 0, 0, 0, 0, 0, -1]
for _ in range(10):
    obs, _, _, _ = env.step(dummy)

print("Step 8: Running inference step...", flush=True)
observation = {**obs, "task_description": task.language}
result = policy.infer(observation)
action = result["actions"]
print(f"Action: {action[:4]}..., shape={action.shape}", flush=True)

print("Step 9: Stepping env with action...", flush=True)
padded = pad_action_for_env(action, "cartesian_pose", 7)
obs, reward, done, info = env.step(padded.tolist() if hasattr(padded, "tolist") else padded)
print(f"reward={reward}, done={done}", flush=True)

env.close()
policy.close()
print("=== DEBUG EVAL PASSED ===", flush=True)
