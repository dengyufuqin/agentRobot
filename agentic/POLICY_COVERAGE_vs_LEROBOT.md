# Policy coverage: AgentRobot vs huggingface/lerobot

Snapshot as of 2026-04-24. Compares what each framework can deploy
end-to-end against a benchmark (not just "the repo has the code").

## Summary

| Dimension | HF LeRobot | AgentRobot (this repo) |
|---|---|---|
| Policies natively deployable | 15 | 9 first-party + all lerobot aliases |
| Benchmarks with adapters | LeRobot sim + real-robot | LIBERO, ManiSkill, RoboCasa, RoboTwin (4) |
| One-sentence deployment | No | Yes (`run_benchmark --policy X --benchmark Y`) |
| Eval-protocol gate before submission | No | Yes (`eval_protocols/*.json` validated) |
| Non-lerobot VLA integration | No | OpenVLA + SpatialVLA + OpenPI-JAX native |
| Agentic orchestration | No | `agent.py` + skill system |

## Policy-by-policy matrix

Source: `lerobot/src/lerobot/policies/` vs `run_benchmark.py::POLICY_CONFIGS`
+ `lerobot/policy_server.py` type-routing.

| Policy | HF LeRobot | Our fork | Server | Notes |
|---|---|---|---|---|
| act | ✓ | via lerobot wrapper | lerobot | |
| diffusion | ✓ | ✓ | diffusion_policy (standalone) | |
| pi0 | ✓ | ✓ | lerobot | verified: LIBERO 64/84/72/34 % |
| pi05 | ✓ | ✓ | lerobot | verified: LIBERO 88/89/87/85 %; ManiSkill PullCube deadlocks via openpi route (see gap) |
| pi0_fast | ✓ | ✓ | lerobot | verified: LIBERO 94/96/86/84 % (`lerobot/pi0fast-libero`); mislabeled `jadechoghari/pi0fast-libero` caught by ckpt-class harness sensor |
| smolvla | ✓ | ✓ | lerobot | |
| groot | ✓ | loader routed | lerobot | NVIDIA ckpts use native HF-transformers config (no `type=groot` key) → needs custom server wrapper |
| xvla | ✓ | loader routed | lerobot | untested |
| wall_x | ✓ | loader routed | lerobot | untested |
| vqbet | ✓ | ✗ | — | not wired |
| tdmpc | ✓ | ✗ | — | offline RL, not wired |
| sac | ✓ | ✗ | — | online RL, not wired |
| sarm | ✓ | ✗ | — | not wired |
| multi_task_dit | ✓ | ✗ | — | not wired |
| rtc | ✓ | ✗ | — | not wired |
| rdt | ✗ | via lerobot alias | lerobot | |
| **openvla** | ✗ | ✓ | openvla (standalone) | base + OFT; verified: LIBERO goal 83%, object 82%, libero_10 62%; ManiSkill PullCube 80% (paper 65%) |
| **spatialvla** | ✗ | ✓ | spatialvla (standalone) | |
| **openpi (JAX)** | ✗ | ✓ | openpi (standalone) | handles RLinf/kimtaey/orbax ckpts |

**Takeaway.** LeRobot covers 6 RL/classical policies (vqbet, tdmpc, sac,
sarm, multi_task_dit, rtc) we don't wire. We cover 3 big non-lerobot VLA
families (OpenVLA, SpatialVLA, OpenPI-JAX) they don't. Intersection on
VLA is essentially equal; the value each side adds is orthogonal.

## Benchmark coverage matrix

LeRobot publishes a single simulated environment scaffold (`lerobot/sim`)
plus real-robot pipelines. AgentRobot is multi-adapter:

| Platform | LeRobot | AgentRobot |
|---|---|---|
| LIBERO (spatial/object/goal/10) | via user glue | native eval client |
| ManiSkill (PickCube/PullCube/...) | — | native eval client |
| RoboCasa (PnP*, TurnOn*, ...) | — | native eval client |
| RoboTwin (beat_block_hammer, open_laptop, ...) | — | native eval client |
| LeRobot sim | ✓ | — |

## What differentiates AgentRobot

1. **One-sentence deployment**: `run_benchmark --policy pi0.5 --benchmark
   libero_10 --submit` from cold → SLURM job in one call.
2. **Protocol gate**: `eval_protocol.json` declares image_resolution,
   max_episode_steps, camera set, flip, center_crop, unnorm_key,
   state_dim/action_dim. Validated before SLURM submission — catches
   misconfig instead of burning a 90-min job on a bad flag.
3. **Per-platform obs adapters**: `libero`, `maniskill`, `robocasa`,
   `robotwin` eval clients auto-translate their native obs dicts to the
   policy-server-expected format (e.g., LIBERO 180° flip, ManiSkill
   HumanCamera wrapper, 8D vs 16D proprio).
4. **Cross-format checkpoint resolution**:
   `resolve_lerobot_nested_ckpt` for `checkpoints/<step>/pretrained_model/`
   layouts; `route_openpi_config` for JAX/orbax ckpts.
5. **Agentic layer**: `agent.py` + `skills/` for LLM-driven benchmark
   design, probe+wrap of unseen GitHub policies, and one-sentence repro.

## Known gaps (documented, prioritized)

- **pi0_fast canonical ckpt** — **resolved 2026-04-24**: the real one is
  `lerobot/pi0fast-libero` (604 tensors, `type=pi0_fast`, no `action_in_proj`
  / `time_mlp` / AdaRMSNorm). The community ckpt `jadechoghari/pi0fast-libero`
  is mislabeled — its safetensors contain 812 pi0.5 tensors.
  Mislabels of this kind are now caught pre-submission by the ckpt-class
  harness sensor (`run_benchmark.py::run_ckpt_compat_gate`): it reads the
  safetensors header via `get_safetensors_metadata` and matches tensor key
  substrings against a per-policy profile (pi0 / pi0.5 / pi0_fast).
- **groot from NVIDIA**: `nvidia/GR00T-N1.5-3B` and community RoboCasa
  fine-tunes publish HF-transformers-native config (no lerobot `type`
  key). Would need a GR00T-specific policy_server using NVIDIA's
  `Isaac-GR00T` loader instead of lerobot's factory.
- **vqbet / tdmpc / sac / sarm / rtc / multi_task_dit**: lerobot-native
  but not wired. Low priority — not the VLA workstream.
- **RoboTwin — no registered ckpts**: the eval_registry has zero RoboTwin
  entries for any policy, and local checkpoint storage has nothing
  RoboTwin-trained. Submissions via `run_benchmark --benchmark robotwin:*`
  are refused at the cross-domain gate (correct behavior — would burn
  compute for ~0% results). RoboTwin adapter + eval client exist; gap is
  a trained checkpoint, not infrastructure.
- **ManiSkill task coverage (RPD ckpts)**: Octo and OpenVLA RPD-release
  ckpts verified on PullCube-v1 (70% / 80%) and PushCube-v1 (40% / 40%).
  Task sweep of PickCube/StackCube/PegInsertionSide scores 0% as expected —
  those tasks are outside the RPD training subset (memory
  `feedback_rpd_task_coverage`). PickSingleYCB crashes on a non-interactive
  asset-download prompt (separate infra gap). See
  `MANISKILL_VERIFIED.md` for the full per-task table.
- **pi0.5 × ManiSkill**: `/.../checkpoints/pi05-maniskill/` has the
  `physical-intelligence/` subdir, so run_benchmark auto-routes to the
  openpi PyTorch server with `--config pi05_libero`. First inference
  deadlocks on cn27 (GPU 100% util, process CPU quiescent) — likely a
  torch.compile-produced kernel choking on the ManiSkill obs shape/dtype
  that a LIBERO-config'd adapter doesn't expect. Needs either (a) a
  ManiSkill-specific openpi config, or (b) a new non-openpi route that
  loads pi0.5-maniskill via lerobot-PyTorch. Triton Python.h fix
  (`feedback_triton_python_h`) is orthogonal — got past that, then
  deadlocked further down.
