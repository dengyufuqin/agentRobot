# π-family × LIBERO reproduction

Snapshot as of 2026-04-24. Each row is one `run_benchmark.py --submit`
invocation, measured end-to-end (protocol gate → SLURM → eval client).

## Results (pass@1, 10 episodes × 10 tasks = 100 rollouts per cell)

| Policy       | spatial | object | goal | 10 (long) | Notes |
|--------------|--------:|-------:|-----:|----------:|-------|
| pi0          |     64% |    72% |  84% |       34% | canonical `lerobot/pi0_libero_finetuned` |
| pi0.5        |     88% |    87% |  89% |       85% | torch.compile JIT ~45 min cached per-node |
| pi0-FAST     |     94% |    96% |  86% |       84% | canonical `lerobot/pi0fast-libero`, 5 ep/task; mislabeled jadechoghari ckpt caught by harness sensor; goal SIGABRT'd on first try, retry passed |
| OpenVLA-OFT  |       — |    82% |  83% |       62% | `Stanford-ILIAD/openvla-7b-oft-libero-…`, spatial blocked |

SLURM job IDs (for `grep -l` in `logs/`):
- pi0:          116681 / 116683 / 116708 / 116684
- pi0.5:        116688 / 116686 / 116709 / 116687
- pi0-FAST:     116757 / 116777 / 116787 / 116779 (goal first try 116778 SIGABRT'd at t=50 of task 0)
- OpenVLA-OFT:   —    / 116711 / 116662 / 116637

## Known gaps

### pi0-FAST — `jadechoghari/pi0fast-libero` mislabel (resolved)

The real pi0-FAST LIBERO ckpt is `lerobot/pi0fast-libero` (604 tensors,
`type=pi0_fast`, no `action_in_proj` / `time_mlp` / AdaRMSNorm). Verified
2026-04-24 at 94% on libero_spatial (job 116757, 5 ep × 10 tasks).

`jadechoghari/pi0fast-libero` is the mislabeled sibling: its safetensors
contain 812 pi0.5-only tensors (`action_in_proj`, `action_out_proj`,
`time_mlp`, `.dense.` AdaRMSNorm). After we worked past three layers
(draccus type lookup → dataclass field set → FAST tokenizer scipy dep),
`PI0FastPolicy.load_state_dict` rejected the weights with `Unexpected
key(s) in state_dict`. First inference then returned uninitialised
`<bos>` tokens (evidence: job 116736 log lines 219-224).

This class of mislabel is now caught pre-submission by the ckpt-class
harness sensor — `run_benchmark.py::run_ckpt_compat_gate` reads the
safetensors header via `get_safetensors_metadata` and matches tensor-key
substrings against `_CKPT_CLASS_MARKERS` (`pi0` / `pi0.5` / `pi0_fast`).
A mislabeled ckpt now fails in ~1 s instead of ~45 min.

### OpenVLA LIBERO-spatial MuJoCo abort

3 consecutive runs (116661, 116710, 116716) reproducibly SIGABRT at t≈50
inside robosuite. Actions grow large (z ≈ 0.26) suggesting physics
instability rather than policy/server bug — pi0 and pi0.5 complete the
same suite on the same EGL pool. Noted under `feedback_egl_nodelist_multi`
as a per-suite numerical-stability issue, not infra.

## Reproduction command

```
python run_benchmark.py \
  --policy {pi0,pi0.5,openvla} \
  --benchmark libero_{spatial,object,goal,10} \
  --submit
```

Protocol gate (`eval_protocols/<policy>_libero.json`) validates
image_resolution / max_episode_steps / camera / flip / center_crop /
unnorm_key / state_dim / action_dim *before* SLURM submission, so a
misconfigured flag does not burn a 90-min H100 job.
