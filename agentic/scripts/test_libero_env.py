#!/usr/bin/env python3
"""Quick test: create LIBERO env, run a few dummy steps, verify rendering."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.utils.run_utils import ARM_CONTROLLER_MAP, enable_joint_pos_observable
import numpy as np

bd = benchmark.get_benchmark_dict()
suite = bd["libero_spatial"]()
task = suite.get_task(0)
init_states = suite.get_task_init_states(0)

bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"Task: {task.language}")
print(f"Creating env...")
env = OffScreenRenderEnv(bddl_file_name=bddl_file, controller="OSC_POSE",
                         camera_heights=256, camera_widths=256)
env.seed(0)
enable_joint_pos_observable(env)
env.reset()
obs = env.set_init_state(init_states[0])

img = obs["agentview_image"]
print(f"agentview shape: {img.shape}, range: [{img.min()}, {img.max()}]")

dummy_action = [0, 0, 0, 0, 0, 0, -1]
for i in range(5):
    obs, r, d, info = env.step(dummy_action)
    if i % 2 == 0:
        print(f"Step {i}: reward={r:.4f}, done={d}")

env.close()
print("=== LIBERO env test PASSED ===")
