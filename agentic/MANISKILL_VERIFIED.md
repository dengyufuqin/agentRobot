# ManiSkill reproduction (PullCube-v1)

Snapshot as of 2026-04-24. Each row is one `run_benchmark.py --submit`
invocation, measured end-to-end (protocol gate → SLURM → eval client).

## Results (pass@1, 10 episodes)

| Policy        | Result | Target | Source                          | Job   |
|---------------|-------:|-------:|---------------------------------|-------|
| Octo          |    70% |    52% | RPD paper (2503.05833) Table 2  | 116796 |
| OpenVLA-base  |    80% |    65% | RPD paper (2503.05833) Table 2  | 116799 |
| pi0.5         |    — (deadlock) | 80% | internal target      | 116800 |

Both verified policies **beat their paper targets** on PullCube-v1.

## Task-coverage sweep (5 episodes each)

Extends the RPD-trained ckpts to four additional ManiSkill tasks to map the
task-coverage boundary. Jobs 116820–116829, submitted via
`run_benchmark --submit` with 5-episode budget.

| Task                | Octo (116820–24) | OpenVLA (116825–29) | Notes |
|---------------------|-----------------:|--------------------:|-------|
| PushCube-v1         | 40% (2/5)        | 40% (2/5)           | In RPD training subset |
| PickCube-v1         | 0%               | 0%                  | NOT in RPD training |
| StackCube-v1        | 0%               | 0%                  | NOT in RPD training |
| PegInsertionSide-v1 | 0%               | 0%                  | NOT in RPD training |
| PickSingleYCB-v1    | crash (EOF)      | crash (EOF)         | Env asset-download prompt blocks in non-interactive SLURM |

**Takeaway.** RPD checkpoints cover only a subset of ManiSkill tasks
(confirmed via memory `feedback_rpd_task_coverage`). PullCube-v1 and
PushCube-v1 are the two known-good tasks; the other three score 0% as
expected. PickSingleYCB needs a non-interactive asset fetch (separate
infrastructure gap, not a policy gap).

## Checkpoint resolution

All three use local ckpts under
`/mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/checkpoints/`:
- `octo-maniskill/` — Octo-base-1.5 finetuned on ManiSkill (RPD release).
  Top-level dir has `config.json`; weights under `60000/default/`.
- `openvla-maniskill/` — OpenVLA-7B finetuned (Juelg, RPD release). Base
  mode, no OFT action-head.
- `pi05-maniskill/` — pi0.5 finetuned (internal). Has
  `physical-intelligence/` subdir that triggers openpi-PyTorch routing.

## Auto-injected flags (from registry + protocol)

The registry's `server_args` and `eval_protocols/*_maniskill.json` are
pulled into the SLURM script by `run_benchmark.py`:
- OpenVLA: `--unnorm_key maniskill_human:7.0.0`, `--num_images_in_input 1`,
  `--no_proprio`, `--no_flip_image`, `--no_center_crop`, `--img_res 256`,
  `--max_episode_steps 300`, `--use_human_camera`, `--no_invert_gripper`.
- Octo: unbounded actions clipped [-1,1] server-side;
  `--use_human_camera`.
- pi0.5 via openpi: `--config pi05_libero`, routed by ckpt heuristic.

## Known gaps

### pi0.5 × ManiSkill — openpi route deadlock

Job 116800 entered Episode 1/10, the server metadata reported
`server_type: openpi_robocasa, config: pi05_libero, action_space:
joint_velocity, action_dim: 8` — then the first `infer()` call
hung. Measured state after 97 min:
- GPU: 100% util, 7.9 GB stable (no memory growth)
- Server python process: 49 s CPU time (idle, not JIT-compiling)
- All 4 torch._inductor.compile_worker processes: <5 s CPU each (idle)
- Log: last line at t+3 s (Episode 1/10 header), nothing after

Diagnosis: a torch.compile-produced CUDA kernel stuck in an infinite loop,
likely because the `pi05_libero` obs adapter output mismatches the kernel's
expected input shape/dtype for ManiSkill's Panda-arm obs dict. Not the
same as LIBERO's legitimate 45-min JIT (memory [feedback_pi05_pytorch_compile_required]).

Fix options (not applied this session):
1. Add a ManiSkill-specific openpi config alongside `pi05_libero`.
2. Route `pi05-maniskill/` to lerobot-PyTorch server instead of openpi, if
   its safetensors format is lerobot-compatible.
3. In `run_benchmark.py`, stop auto-detecting openpi for pi0.5 ckpts whose
   parent benchmark isn't LIBERO — fall back to lerobot server.

Triton `Python.h` gap (`feedback_triton_python_h`) was resolved during
this session by injecting `CPATH`/`C_INCLUDE_PATH` to UV's python include
dir — applies to any torch.compile policy in the openpi venv, not just
this one combo.

## Reproduction command

```
python run_benchmark.py \
  --policy {octo,openvla,pi0.5} \
  --benchmark maniskill:PullCube-v1 \
  --num_trials 10 \
  --submit
```

Protocol gate validates eval_protocols/<policy>_maniskill.json before
SLURM submission; registry lookups auto-inject per-policy flags so the
one-liner above is the complete invocation.
