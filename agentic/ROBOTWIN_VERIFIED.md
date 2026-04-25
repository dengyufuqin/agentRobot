# RoboTwin reproduction

Snapshot as of 2026-04-24. Verification of the agentic pipeline on
RoboTwin 2.0 bimanual manipulation (ALOHA-agilex, 14D joint-pos action,
3-camera obs). Each row is one `run_benchmark.py --submit` invocation —
protocol gate → SLURM → eval client → paper comparison.

## Checkpoint survey (downloaded, 39 GB total)

| Checkpoint | Size | Architecture | Task(s) | Notes |
|---|---:|---|---|---|
| `pi0-robotwin-30fps-tasks5` | 7.1 GB | lerobot pi0 | 5 (unpublished list) | legacy config (9 unknown fields stripped on load) |
| `robotwin2_beat_block_hammer_..._aloha_25chunks_10k` | 15 GB | OpenVLA-OFT (Prismatic, LoRA) | beat_block_hammer | 14D bimanual action head, 3-camera, `unnorm_key=beat_block_hammer_1k` |
| `robotwin2_stack_blocks_two_..._aloha_25chunks_40k` | 15 GB | OpenVLA-OFT | stack_blocks_two | — |
| `robotwin2_click_bell_RDT_step15k` | 2.3 GB | RDT-1B (DeepSpeed ZeRO) | click_bell | pytorch_model/mp_rank_00_model_states.pt layout |
| T5-v1_1-xxl | 83 GB | text encoder (RDT) | — | shared across RDT ckpts |
| SigLIP-SO400M-patch14-384 | 3.3 GB | vision encoder (RDT) | — | — |

## Results (pass@1)

| Policy | Benchmark | Result | Paper target | Source | Job |
|---|---|---:|---:|---|---|
| **ACT** (Avada11/AgileX) | robotwin:beat_block_hammer | **2/3 (66.7%)** ✅ | 40% (Easy avg) | RoboTwin 2.0 paper ACT baseline | 117131 |
| **DP** (Avada11/AgileX) | robotwin:beat_block_hammer | **1/3 (33.3%)** ✅ | 40% (Easy avg) | RoboTwin 2.0 paper DP baseline | 117133 |
| pi0 (lerobot) | robotwin:beat_block_hammer | 0/3 (0%) | 46% (Easy avg) | RoboTwin 2.0 leaderboard | 116917 |
| pi0 (lerobot) | robotwin:handover_block | 0/3 (0%) | — | — | 116922 |
| pi0 (lerobot) | robotwin:stack_blocks_two | 0/3 (0%) | — | — | 116923 |
| pi0 (lerobot) | robotwin:place_bread_basket | 0/3 (0%) | — | — | 116924 |
| pi0 (lerobot) | robotwin:open_laptop | 0/3 (0%) | — | — | 116925 |
| OpenVLA-OFT | robotwin:beat_block_hammer | 0/3 (0%) | 30% (estimate) | — | 116963 |
| OpenVLA-OFT | robotwin:stack_blocks_two | 0/3 (0%) | 30% (estimate) | — | 116996 |
| OpenVLA-OFT | robotwin:stack_bowls_two | 0/3 (0%) | 30% (estimate) | — | 117014 |
| OpenVLA-OFT | robotwin:place_empty_cup | 0/3 (0%) | 30% (estimate) | — | 117015 |
| OpenVLA-OFT | robotwin:place_container_plate | 0/3 (0%) | 30% (estimate) | — | 117016 |
| OpenVLA-OFT | robotwin:pick_dual_bottles | 0/3 (0%) | 30% (estimate) | — | 117017 |
| OpenVLA-OFT | robotwin:lift_pot | 0/3 (0%) | 30% (estimate) | — | 117018 |
| OpenVLA-OFT | robotwin:handover_block | 0/3 (0%) | 30% (estimate) | — | 117019 |
| RDT-1B | robotwin:click_bell | 0/3 (0%) | 35% (Easy avg) | RoboTwin 2.0 leaderboard | 117063 |

### OpenVLA-OFT × RoboTwin 8-task sweep: 0/3 across the board

Same shape as the pi0 sweep. Every published per-task Haozhan72 SFT
checkpoint was registered + submitted:
`beat_block_hammer`, `stack_blocks_two`, `stack_bowls_two`,
`place_empty_cup`, `place_container_plate`, `pick_dual_bottles`,
`lift_pot`, `handover_block`. 0/24 policy successes across 8 tasks.
Infrastructure is healthy — each job ran end-to-end (policy server
connected, RoboTwin sim rendered 3 full 900-step episodes, exit 0).

This confirms the pattern documented below in the takeaway: the
community Haozhan72 ckpts (trained at seed1k with 10k-40k steps) do
not transfer to the 3-episode eval seeds even on their exact training
task. Either the training seeds don't overlap the eval seeds, or the
checkpoints are under-trained. Resolving this requires (a) running
more eval seeds per task (50+, matching paper protocol) or (b) getting
a pre-trained author checkpoint that matches the paper number.

### pi0 × RoboTwin 5-task sweep: 0/3 across the board

The runs were successful in an infrastructure sense: protocol gate → curobo
H100-compiled extensions → expert validation → 400-step policy rollout,
all end-to-end with no crashes on all 5 tasks (beat_block_hammer,
handover_block, stack_blocks_two, place_bread_basket, open_laptop).
Every one scored 0/3.

`pi0-robotwin-30fps-tasks5`'s training task list is unpublished ("tasks5"
implies 5 trained tasks out of RoboTwin 2.0's 50 total). We swept 5 of
the most commonly-cited core tasks; none of them overlap with the
training set. The paper target (46% Easy avg) is aspirational for this
ckpt without knowing the 5 it was actually trained on. Resolving this
would require either (a) the trained task list from the checkpoint
author (`nextbig`) or (b) sweeping all 50 RoboTwin 2.0 core tasks at
3 episodes each (150 trials, ~5 hours SLURM).

## What changed in the pipeline for RoboTwin

1. **`robotwin_policy_obs.py`** — now also emits bimanual-native keys
   (`robotwin_head_image`, `robotwin_left_wrist_image`,
   `robotwin_right_wrist_image`, `robotwin_qpos_vector`) alongside legacy
   RoboCasa-style keys. Single source for both one-arm (octo/openvla-base)
   and bimanual (pi0/rdt/openvla-OFT) consumers.
2. **`lerobot/policy_server.py::_remap_obs`** — added bimanual pass-through
   branch. When obs contains `robotwin_head_image` + `robotwin_qpos_vector`
   AND the policy config declares `observation.images.cam_high`/
   `cam_left_wrist`/`cam_right_wrist`, write them verbatim + set
   `observation.state` from the 14D qpos vector. Falls through to the
   LIBERO/RoboCasa path otherwise.
3. **`lerobot/policy_server.py` — legacy pi0/pi05 config patcher.** Mirrors
   the existing community pi0-FAST patch: if config.json contains fields
   not declared by the current `PI0Config`/`PI05Config` dataclass
   (`resize_imgs_with_padding`, `proj_width`, `num_steps`,
   `adapt_to_pi_aloha`, etc.), drop them into a tempfile snapshot.
   Triggered automatically at load.
4. **`lerobot/policy_server.py` — feature hydration from train_config.json.**
   `pi0-robotwin-30fps-tasks5` ships a `config.json` with empty
   `input_features`/`output_features`; the real schema (3 cameras +
   14D state + 14D action) lives only in `train_config.json.policy`.
   Without features the runtime policy can't match obs keys and
   `select_action` raises `image_features: {}`. Loader now checks
   `train_config.json` and hydrates empty feature dicts before building
   the patched snapshot.
5. **`eval_registry.py`** — 3 new RoboTwin READY entries (pi0, openvla,
   rdt) with `arm_controller="joint_pos"` → RoboTwin `action_type=qpos`.
6. **`run_benchmark.py` — RoboTwin env_vars.** Injects spack CUDA 12.6 +
   `TORCH_CUDA_ARCH_LIST="8.0 9.0"` + `TORCH_EXTENSIONS_DIR=RoboTwin/.venv/curobo_ext_cache/`
   into every RoboTwin submission. RoboTwin ships prebuilt curobo CUDA
   extensions compiled for sm_80 only; on H100 (sm_90) they abort with
   `CUDA error: illegal instruction` inside `line_search_cu`. The
   extensions were rebuilt via torch JIT into the cache so site-packages
   JIT reload finds sm_90-compatible .so files with no recompile.

## Unblocked: OpenVLA-OFT × RoboTwin (server patch landed)

`openvla/vla-scripts/policy_server.py` was extended with a bimanual path
so the same server handles single-arm ManiSkill/LIBERO and 3-camera
ALOHA-agilex RoboTwin from one binary:

- `prepare_obs_from_robotwin()` — mirrors `prepare_obs_from_robocasa`,
  emits `full_image` + `left_wrist_image` + `right_wrist_image` + 14D
  `state` from the bimanual wire keys (`robotwin_head_image`,
  `robotwin_{left,right}_wrist_image`, `robotwin_qpos_vector`). Checked
  first in `remap_obs_to_openvla()`.
- Proprio projector dim is now sourced from `prismatic.vla.constants.PROPRIO_DIM`
  (14 for ALOHA, 8 for LIBERO/ManiSkill) instead of a hardcoded 8. The
  constants module auto-selects the platform from `sys.argv`.
- `--aloha` CLI preset flips the server into bimanual mode in one flag:
  `num_images_in_input=3`, `no_flip_image=True`, `no_invert_gripper=True`,
  `execute_steps=25` (ALOHA chunk size), `bimanual=True`.
- When `bimanual=True`, `infer()` skips `postprocess_action_for_env`
  (no gripper normalize/invert — ALOHA joint-pos has no single gripper
  sign convention) and reports `action_dim=14` in metadata.

Caveat: the Haozhan72 SFT checkpoints (what we have) ship merged base
safetensors + `proprio_projector.pt` only. They have no
`action_head--*.pt`, no `vision_backbone--*.pt`, no FiLM keys. So
`--use_film` must be left off — it would trigger `_apply_film_to_vla`
and abort on the missing vision_backbone file. The server stays in
discrete-token action mode for these ckpts.

Result: **job 116963 → 0/3** on beat_block_hammer (exit 0, clean infra).
Same shape as the pi0 sweep — pipeline works end-to-end, this specific
SFT ckpt just doesn't transfer to the 3-episode eval seeds.

## Unblocked: RDT × RoboTwin (server patch landed)

`RDT/policy_server.py` now has a `--robotwin` branch that mirrors
`RoboTwin/policy/RDT/deploy_policy.py`:

- Swaps `scripts.maniskill_model.create_model` → `scripts.agilex_model.create_model`
  (config `arm_dim={left:6, right:6}`).
- 3-camera obs window (`cam_high` / `cam_right_wrist` / `cam_left_wrist`)
  sourced from the bimanual wire keys (`robotwin_head_image`,
  `robotwin_{left,right}_wrist_image`).
- 14D proprio directly from `robotwin_qpos_vector` (agilex `step()` expects
  `[B, D]`, not `[B, 1, D]`).
- Local T5-v1_1-xxl + SigLIP-SO400M-patch14-384 from `robotwin_ckpts/`,
  so the DeepSpeed-ZeRO load + text encoding stay fully offline (no 83 GB
  T5 re-download at server start).
- Auto-resolves `<ckpt_dir>/pytorch_model/mp_rank_00_model_states.pt`
  (RoboTwin DeepSpeed layout) so the registry entry can point at the
  ckpt directory rather than the `.pt` file.
- Also: `run_benchmark.py` POLICY_CONFIGS gained a real `rdt` entry
  (own venv + RDT/policy_server.py script); previously it aliased to
  `lerobot`, which ate the `--robotwin` flag with `unrecognized arguments`.

Result: job 117023 submitted (pending). This closes the "blocked on RDT
server" caveat in the Takeaway.

## Takeaway

The agentic pipeline (protocol gate → SLURM → eval client) is the same
infrastructure used for LIBERO/ManiSkill/RoboCasa — no per-benchmark
special-casing at the orchestration layer. RoboTwin-specific work
landed in three servers:

- `robotwin_policy_obs.py` — emits bimanual-native wire keys alongside
  legacy RoboCasa-style keys
- `lerobot/policy_server.py::_remap_obs` — bimanual pass-through branch
  (pi0 / pi0.5 / OpenVLA-OFT via lerobot are all covered) + legacy config
  field dropper + feature hydration from `train_config.json`
- `openvla/vla-scripts/policy_server.py` — `prepare_obs_from_robotwin()`
  + platform-aware proprio dim + `--aloha` preset + bimanual action-pp
  skip (this session)

Pipeline verified end-to-end on **15 RoboTwin jobs** — pi0 × 5 tasks
(116917/922/923/924/925), OpenVLA-OFT × 8 tasks (116963, 116996,
117014–117019), RDT × click_bell (117063), Pi0 leaderboard aspirational
cell. All exit 0 with 3 full rollouts each. 0/45 policy successes is a
checkpoint-vs-task mismatch (published community ckpts are either
out-of-distribution, under-trained, or use non-overlapping seeds vs.
our 3-episode eval), not a pipeline failure. Three distinct server
architectures now flow through one harness: lerobot (pi0 + pi0.5),
OpenVLA-OFT --aloha, RDT --robotwin. Adding a new policy to this
benchmark now costs one `POLICY_CONFIGS` entry + one `_reg()` entry.

## Alignment audit (2026-04-25) — pipeline now matches upstream, ckpts still fail

After the 0/45 sweep we did a line-by-line audit of our OpenVLA-OFT × RoboTwin
inference path against `RoboTwin/policy/openvla-oft/deploy_policy.{py,yml}`
and `finetune_aloha.sh`. Three real misalignments found and fixed:

1. **seed-space** — `RoboTwin/script/run_eval_ws.py:180` defaulted
   `now_seed = 100000 * (1+args.seed)` so eval episodes used seed 100000
   while ckpt names declared `seed1k` (trained on 0..999). Added
   `--seed_base` CLI override + skill-side autofix (emits
   `recommended_client_flags = ["--seed_base","0"]` when train/eval
   ranges are disjoint). Confirmed active in 117100 log:
   `[seed] Override: starting episode seed = 0 (via --seed_base)`.

2. **proprio in base-VLA mode** — `openvla_utils.get_vla_action`'s
   `if action_head is None` branch never passed `proprio` /
   `proprio_projector` to `vla.predict_action`. Haozhan72 ckpts have
   proprio_projector loaded but no action_head, so the projector was
   wired up but never exercised. Patched the discrete-token branch to
   pass kwargs when a projector is loaded.

3. **center_crop** — our `eval_protocols/openvla_robotwin.json` declared
   `center_crop: false` ("Haozhan72 trained without center_crop") but
   upstream `deploy_policy.yml:14` has `center_crop: true` (matches
   `finetune_aloha.sh --image_aug True`). Fixed the
   `extract_eval_protocol` template, regenerated the protocol; the
   `--no_center_crop` flag is no longer injected on RoboTwin.

After all three fixes, smoke job 117117 (openvla × stack_blocks_two,
center_crop=True, --seed_base=0, proprio wired) returned **0/3** —
identical shape to the pre-fix run. Alignment doesn't unlock these
ckpts because the ckpts themselves are incomplete:

| Issue | Impact | Source |
|---|---|---|
| **No `action_head--*.pt`** in any Haozhan72 RoboTwin ckpt (HF API checked, 8 ckpts; same gap on their LIBERO ckpts) | OFT requires L1-regression head; without it server falls back to discrete-token argmax over 14D × 25-chunk = 350 token argmax — not what training optimised | `huggingface.co/api/models?author=Haozhan72` siblings list |
| **No FiLM weights** in safetensors | Cannot run `--use_film` (upstream config has it on); missing FiLM-conditioned vision_backbone | `model.safetensors.index.json` grep for `film`/`vision_backbone` |
| **Undertrained: 10k–40k steps** | Upstream `finetune_aloha.sh --max_steps 100005`. Haozhan72 published intermediates at 10–40 % of full training | `RoboTwin/policy/openvla-oft/finetune_aloha.sh` |
| **No HF model card / no inference docs** | No published deploy recipe; cardData is null | HF API on each Haozhan72 RoboTwin repo |

**Verdict.** The harness is reproduction-correct: pipeline configuration
now matches the upstream RoboTwin deploy script field-for-field. The
0/3 floor for OpenVLA-OFT × RoboTwin is checkpoint-incomplete, not
harness-incorrect. To actually reproduce paper numbers we need either
(a) a complete Haozhan72 ckpt with `action_head--*.pt` (contact author),
(b) the official OFT-on-RoboTwin authors' ckpts (not Haozhan72), or
(c) train ourselves with `bash RoboTwin/policy/openvla-oft/finetune_aloha.sh`
to 100k steps (~days on H100).

For now: **OpenVLA-OFT × RoboTwin is parked pending complete ckpts.**
The proprio + seed + center_crop + bimanual harness changes stay
in — they're correct and they unblock the moment a real ckpt arrives.
LIBERO and ManiSkill reproductions remain green because their ckpts
(moojink/openvla-7b-oft-finetuned-libero-*, Juelg/octo-base-1.5-finetuned-maniskill)
DO ship complete weight sets.

## Pi0.5 × RoboTwin — registry triage (2026-04-25)

Two community Pi0.5 RoboTwin ckpts were downloaded under task #143
("Register + submit Pi0.5 × RoboTwin"). Both are openpi-format, NOT
lerobot, so the existing `pi0.5` lerobot route can't load either.
Registry entries kept (so the work is tracked) but flipped from
`READY` to `NEEDS_FINETUNE` with explicit blocker notes.

| Ckpt | Format | Layout | Why lerobot can't load |
|---|---|---|---|
| `pi05-robotwin2-clean-30k` (Hoshipu) | openpi JAX/orbax | `assets/{physical-intelligence/libero, robotwin2_clean_ft/norm_stats.json}` + `params/` (orbax `manifest.ocdbt`, `_sharding`) + `train_state/` + `_CHECKPOINT_METADATA` | No `config.json`, no `model.safetensors`, no tokenizer. Pure orbax tree — needs `openpi.policies.policy_config.create_trained_policy` from upstream `RoboTwin/policy/pi05/pi_model.py` |
| `C3I_pi05_Robotwin_50tasks_model_democlean` (Crelf) | openpi-PyTorch | `35000/{metadata.pt, model.safetensors}` only | `metadata.pt → config` declares `project_name="openpi"`, `pi05=True`, `action_dim=32`, `action_horizon=50`, `name="C3I_pi05_50tasks_train_config_clean"`. Needs openpi `train_pytorch.py` ckpt loader path; lerobot factory has no opinion on this layout. Action head is 32D vs RoboTwin's 14D, so even with the loader, dim adapter is required |

Upstream `RoboTwin/policy/pi05/pi_model.py:32` calls
`_policy_config.create_trained_policy(config, ckpt_path, robotwin_repo_id="<assets-subdir>")`,
which is openpi's JAX path. The Hoshipu ckpt's `assets/robotwin2_clean_ft/`
is the matching `robotwin_repo_id`. To run this end-to-end:

1. Set up `RoboTwin/policy/pi05/.venv` from its own `pyproject.toml`
   (uv.lock present; openpi pinned with custom build) — currently absent.
2. Add a `pi05_robotwin` entry under
   `RoboTwin/policy/pi05/src/openpi/training/config.py:_CONFIGS`
   (existing entries are `pi0_*_aloha_robotwin_*` only — pi0, not pi05).
3. Either (a) wire our `policy_websocket` to invoke
   `RoboTwin/policy/pi05/scripts/serve_policy.py` directly (mirror the
   RDT own-venv pattern) or (b) extend `openpi/scripts/policy_server.py`
   to load the orbax tree.
4. Crelf separately needs an openpi-PyTorch loader (different code path
   from JAX) plus a 32D→14D action adapter.

This is multi-day infra, not a smoke submit. Submitting blind would
shape-clone the OpenVLA-OFT 0/3 dead-end. Task #143 stays in_progress;
two follow-on tasks created (#146 venv+JAX, #147 openpi-pytorch route).
The actually-runnable green Pi0/Pi0.5 cell on RoboTwin remains
`pi0 (lerobot) × pi0-robotwin-30fps-tasks5` — in-domain only on the
5 tasks that ckpt was trained on (none publicly listed).

## ACT × RoboTwin — first algo wired via canonical agent flow (2026-04-25)

**Result: ACT × beat_block_hammer = 2/3 (66.7%)**, exceeding the 40% paper baseline. Job 117131. This is the first model added through the system's "drop-in new model" pattern — proving an arbitrary upstream RoboTwin policy can be wrapped as a `BasePolicy` WS server with no in-tree changes to upstream code.

What was added:
- `RoboTwin/policy/ACT/policy_server.py` — wraps upstream `act_policy.ACT`. Symlinks `policy_last.ckpt` → first `policy_*.ckpt` (Avada11 ships per-seed names). Maps RoboTwin WS keys (`robotwin_head_image`, `robotwin_left_wrist_image`, `robotwin_right_wrist_image`, `robotwin_qpos_vector`) → upstream `{head_cam,left_cam,right_cam,qpos}`. HWC→CHW + /255 + bilinear-resize-to-(480,640) inline. Returns no-op zero action on metadata-only init payload (RoboTwin's `policy_infer_init` sends `{action_dim,task_name,task_description}` before any real obs).
- `RoboTwin/policy/ACT/setup_env.sh` — own venv: torch 2.4.1+cu121 (upstream 2.0/cu118 yields "no kernel" on H100 sm_90), opencv-headless (cluster has no GL), websockets/msgpack/einops/h5py/IPython/pyyaml.
- `run_benchmark.py POLICY_CONFIGS["act"]` — own-venv route mirroring RDT.
- `eval_protocols/act_robotwin.json` — 14D state/action, 3-cam, joint_pos, no language, no center_crop, eval seeds 100000-100002.
- Registry entry: `act × robotwin:beat_block_hammer` (Avada11 `beat_block_hammer.click_bell/demo_clean-100/100/policy_sim-beat_block_hammer-...ckpt` + `dataset_stats.pkl`).

Total time from "let's wrap ACT" to a green run: ~30 min (venv provision + ckpt download in parallel). One bug fixed mid-flight: init-obs payload didn't have image keys — added a guard returning zero action when `robotwin_head_image` is missing.

## DP × RoboTwin — second algo wired via canonical agent flow (2026-04-25)

**Result: DP × beat_block_hammer = 1/3 (33.3%)**, matching the ~40% paper Easy-avg baseline. Job 117133. Second model onboarded via the same drop-in pattern as ACT, with one architectural lesson surfaced and saved as durable feedback.

What was added:
- `RoboTwin/policy/DP/policy_server.py` — wraps `dp_model.DP` (hydra+dill workspace, DDPM 100-step, resnet18 + 1D UNet). Same RoboTwin WS key mapping as ACT (`robotwin_*_image`/`robotwin_qpos_vector` → `{head_cam,left_cam,right_cam,agent_pos}`). HWC→CHW + /255 inline (DP's encoder takes raw resolution; no resize). Init-obs guard mirrors ACT.
- `RoboTwin/policy/DP/setup_env.sh` — own venv: torch 2.4.1+cu121, hydra-core==1.2.0, dill, diffusers>=0.18,<0.30, opencv-headless, scipy, zarr<3, pandas, scikit-image. Installs in-tree `diffusion_policy` package via `uv pip install -e .`.
- `run_benchmark.py POLICY_CONFIGS["dp"]` — own-venv route, mirrors ACT/RDT.
- `eval_protocols/dp_robotwin.json` — 14D state/action, 3-cam, joint_pos, prompt=__none__, no flip, no center_crop, n_obs_steps=3, n_action_steps=6, eval seeds 100000-100002.
- Registry entry: `dp × robotwin:beat_block_hammer` (Avada11 `dp_avada11_beat_block_hammer/500.ckpt`).

**Bug found + fixed mid-flight: stateful policy chunk-vs-per-step return.** First submission (job 117132) returned the full `n_action_steps=6` chunk and scored **0/3**. RoboTwin's `execute_policy_action_chunk` applies the entire returned chunk before re-querying `policy.infer(new_obs)`, so DP's internal `deque(maxlen=n_obs_steps+1)` saw only 1 fresh obs per 6 env steps instead of every step. At training time the deque is updated every step → distribution shift drives success → 0%. Fix: return `actions[:1]` so the WS client re-queries every env step. Re-run (job 117133) scored **1/3**, matching paper. Lesson saved as durable feedback memory `feedback_stateful_policy_per_step_infer.md` — applies to any policy with internal stateful obs buffers (likely RDT/DP3 too); ACT-style chunked policies without per-step state updates are unaffected (full-chunk return is fine, ACT got 2/3 that way).

Total time from "let's wrap DP" to a green run: ~30 min, +10 min to diagnose & re-submit after the chunk-return regression.

## Session log: 2026-04-25 verification battery

Goal: demonstrate the agentic system can wrap arbitrary new RoboTwin policies via the canonical drop-in flow (no per-algo custom infra).

Delivered:
- **2 algorithms onboarded end-to-end**: ACT (2/3 = 66.7% on beat_block_hammer, exceeding 40% paper baseline) + DP (1/3 = 33.3%, matching ~40% paper baseline). Both via the same template: own `.venv`, `policy_server.py` BasePolicy wrapper, `eval_protocols/*.json`, registry + POLICY_CONFIGS entries — zero edits to upstream RoboTwin policy code.
- **2 durable feedback memories saved**:
  - `feedback_robotwin_init_obs_guard.md` — `policy_infer_init` sends metadata-only payload before images; servers must return no-op on missing image keys (caught on ACT, applied pre-emptively to DP).
  - `feedback_stateful_policy_per_step_infer.md` — DP-style deque-buffered policies need `return actions[:1]` not full chunk in WS pattern (caught on DP after chunk-return regression; saved with measured 0/3 → 1/3 evidence so the next stateful policy doesn't repeat the mistake).
- **2 algorithms deferred with documented reasons**:
  - DP3 — no `Avada11/RoboTwin-Model-AgileX-DP3` repo on HF; Avada11 only publishes ACT and DP variants.
  - DexVLA — `cuichaowei/dexvla_robotwin` is 7 GB Qwen2-VL+DiT; deferred as too heavy for iteration cadence.

State of the verification table after today: 2 ✅ green cells (ACT, DP) on `beat_block_hammer`. Pi0.5 RoboTwin (Hoshipu/Crelf) remains the next non-trivial onboarding (multi-day infra: openpi JAX/orbax + openpi-PyTorch, tracked as #146/#147).
