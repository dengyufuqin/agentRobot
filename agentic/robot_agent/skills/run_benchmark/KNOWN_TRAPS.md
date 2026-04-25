# run_benchmark — known traps

Per-skill harness doc. Each trap is something that burned real SLURM time
in prior runs. If you are about to run this skill on a new combo, skim the
relevant section below first — almost every entry here was added *after* a
0% or stalled job.

Back-references (`[mem:…]`) point at the full-context memory entry under
`~/.claude/projects/-mnt-vast-home-yd66byne-code-agentRobot/memory/`.

---

## Infra / SLURM

### EGL nodelist must list all live nodes, not one pick
**Symptom:** 7 LIBERO/RoboCasa jobs queue against one node (e.g. cn26) while
peer EGL nodes sit idle.
**Fix:** `_pick_available_egl_node` returns a comma-separated list of every
currently-`idle`/`mix` EGL node; SLURM spreads. Do not revert to a single
random pick. [mem:feedback_egl_nodelist_single_vs_multi]

### Default port collides under concurrent submission
**Symptom:** Second and later co-located jobs crash with
`OSError: [Errno 98] Address already in use`.
**Fix:** When `args.port == 18800` (argparse default), `submit_as_slurm_job`
randomizes to `[19000, 29999]`. [mem:feedback_port_collision_concurrent_slurm]

### ManiSkill stalls silently on cn16/cn17/cn19
**Symptom:** "Episode 1/10" prints then 11–51 min of silence; peer nodes
complete the same job in 2–3 min.
**Fix:** For ManiSkill submissions, add `cn16,cn17,cn19` to
`#SBATCH --exclude`. Per-node sapien/Vulkan bug, not fixable from our side.
[mem:feedback_maniskill_node_stalls]

---

## Checkpoint resolution

### LeRobot local training layout is nested
**Symptom:** `draccus.utils.ParsingError: Expected a dict with a 'type' key`
when passing an outer training-output dir.
**Fix:** `resolve_lerobot_nested_ckpt` auto-picks
`<ckpt_dir>/checkpoints/last/pretrained_model/` (or latest step). Pass that
nested path to the policy server. [mem:feedback_lerobot_nested_ckpt_layout]

### No silent cross-domain fallback
**Symptom (pre-2026-04-24):** User asks for `pi0 × robocasa`, registry has
no match, run_benchmark silently picks a LIBERO ckpt → 0%.
**Fix:** `_LIBERO_DEFAULTS` only applies when `bench_short == "libero"`.
Other domains hard-exit with a "train one first" message. Do not re-enable
cross-domain defaults. [mem:feedback_system_improvements — principle]

### Ckpt class must match policy loader (safetensors tensor-key preflight)
**Symptom:** `Unexpected key(s) in state_dict` at first inference after
~45 min of JIT compile — pi0_fast loader fed pi0.5 weights.
**Fix:** `run_ckpt_compat_gate` reads the safetensors header via
`huggingface_hub.get_safetensors_metadata` and matches tensor-key
substrings against `_CKPT_CLASS_MARKERS`. `jadechoghari/pi0fast-libero` is
the canonical mislabeled ckpt caught this way.
[mem:feedback_pi0_fast_libero_mislabeled]

---

## Protocol gate

### Protocol gate must key on user-facing policy name, not post-route
**Symptom:** Gate looks up `openpi_maniskill.json` instead of
`pi0.5_maniskill.json` → always-fail on auto-routed openpi ckpts.
**Fix:** `user_policy_name = args.policy` is captured *before* the openpi
auto-route block. [mem:feedback_protocol_gate_routing]

### Protocol validation is not sufficient — fields must be injected as flags
**Symptom:** `eval_protocol.json` passes the gate, but the SLURM script
still runs with stale defaults (e.g. `max_episode_steps=220` for a 300-step
task, or 256×256 when the protocol says 224×224). 0% result.
**Fix:** run_benchmark translates validated protocol fields into concrete
`--flags` on both client and server: `max_episode_steps`, `img_res`,
`camera`, `flip`, `center_crop`. [mem:feedback_protocol_flag_injection]

### Shape fields (state_dim / action_dim / required_obs_keys / obs_wrapper)
The protocol gate checks these too, pre-submission. pi0fast trained
dual-arm (state_dim=16) vs a RoboCasa client sending 8D would previously
crash at first inference — now it is caught at gate time.
[mem:feedback_extract_eval_protocol_shape_fields]

---

## Policy-specific

### LeRobot pi0/pi0.5/smolvla on LIBERO need 180° image flip
**Symptom:** Model loads, action scale looks correct, 0% success.
**Fix:** In `lerobot/policy_server.py`, when ckpt name contains "libero"
apply `img[:, ::-1, ::-1].copy()` before inference. Baked into the LIBERO
training data, not into the policy preprocessor.
[mem:feedback_lerobot_libero_flip]

### OpenVLA on ManiSkill must NOT flip
**Symptom:** HumanCamera render is upright, but server's LIBERO/RoboCasa
branch auto-flips → 0%.
**Fix:** Pass `--no_flip_image`. The LIBERO/RoboCasa flip is unconditional
otherwise. [mem:feedback_openvla_maniskill_no_flip]

### OpenVLA center_crop must match training
**Symptom:** PullCube 40% vs paper 92%; every other field was correct.
**Fix:** `--no_center_crop` for Juelg/RPD ManiSkill ckpts (trained without
crop aug). Default `center_crop=True` is wrong for this family.
[mem:feedback_openvla_center_crop]

### OpenVLA OFT vs base — inference cost is a 10-100× cliff
**Symptom:** Base-mode OpenVLA eval stalls past the 2 h SLURM limit on
episode 1.
**Fix:** Detect `action_head.pt` in the ckpt; if present, use
`--use_l1_regression` (OFT path, one forward per action). Otherwise accept
the long runtime or pick a different policy.
[mem:feedback_openvla_oft_vs_base]

### RPD OpenVLA-ManiSkill is base-VLA only
Specific case of the above: `Juelg/openvla-7b-finetuned-maniskill` ships
no action_head. Must pass `--no_flip_image`, `--no_invert_gripper`,
`--img_res 224`. ~40% on PullCube after all fixes.
[mem:feedback_rpd_openvla_base_mode_scale]

### RPD checkpoints cover a narrow task set — PickCube is NOT trained
**Symptom:** RPD Octo/OpenVLA score 0/10 on `PickCube-v1` even with every
other fix; same ckpts score 8/10 on `PullCube-v1`.
**Fix:** Don't assume coverage. Consult the RPD dataset before picking a
task. `PullCube-v1` is the known-good default.
[mem:feedback_rpd_task_coverage]

### Octo/OpenVLA ManiSkill need HumanCameraWrapper
**Symptom:** Base-camera rendering → 0% even with every fix above; the
training data is 3rd-person human-view.
**Fix:** Wrap env so `env.render()` overwrites
`obs["sensor_data"]["base_camera"]["rgb"]`. Pass `--use_human_camera`.
[mem:feedback_octo_maniskill_human_camera]

### OpenPI LIBERO config uses LIBERO obs keys, not DROID
**Symptom:** pi05_libero crashes or zero-ins on missing
`observation/exterior_image_1_left`.
**Fix:** `openpi/scripts/policy_server.py` must route by config name:
`pi05_libero` → `LiberoInputs` (observation/image, observation/wrist_image,
observation/state 8D, prompt). DROID adapter only applies to
pi05_droid / pi0_fast_droid / pi0_droid. [mem:feedback_openpi_libero_vs_droid_obs]

### pi0.5 PyTorch path requires torch.compile
**Symptom:** First inference = 45+ min of Triton autotune; disabling
compile makes inference silently stall at 100% GPU with no per-step output.
**Fix:** Accept the one-time compile, cache per node, do not promise fast
verify. Use `ping_interval=None` on the websocket client so the compile
doesn't kill the connection. [mem:feedback_pi05_pytorch_compile_required,
feedback_websocket_jit_timeout]

### SmolVLA / pre-migration VLM ckpts need in-memory processor build
**Symptom:** `self._preprocessor=None` after factory fallback; model hits
`KeyError: observation.language.tokens` because TokenizerProcessorStep
never ran.
**Fix:** For any lerobot VLM, if `make_pre_post_processors` fails, build
processors in memory via `make_<type>_pre_post_processors` and extract
buffer stats from `model.safetensors`. Never leave preprocessor=None.
[mem:feedback_smolvla_lerobot_server_gaps]

---

## Agentic path

### LeRobot arm_controller default was joint_vel (wrong for pi0/pi0.5 LIBERO)
Already fixed: `POLICY_CONFIGS["lerobot"]["arm_controller"] = "cartesian_pose"`,
plus registry auto-override. Do not re-default to joint_vel.
[mem:feedback_agentic_path_bugs]

### OpenVLA unnorm_key needs per-suite mapping
`libero_spatial_no_noops` was hardcoded as server default — wrong for
goal/object/10/90. Registry provides per-suite unnorm_key.
[mem:feedback_agentic_path_bugs]
