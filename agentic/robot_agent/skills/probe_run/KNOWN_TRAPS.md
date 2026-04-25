# probe_run — known traps

Per-skill harness doc. The shape-capture hook mechanism has three
failure modes that have each burned at least one job — read before
adding hooks to a new policy.

Back-references (`[mem:…]`) point at the full memory entry.

---

### Success markers must fire AFTER a real forward()
**Symptom:** `io_spec_hook=true` succeeds on a pre-forward log line
("Warming up model", "JIT compiling"), SIGTERM kills the process before
the hook runs, shapes are never captured.
**Fix:** Do not add pre-forward log lines to `success_markers`.
probe_run auto-prepends `[PROBE-HOOK] captured` when
`io_spec_hook=true` — which fires from the hook itself, after a real
forward. Let that be the terminator. [mem:feedback_probe_run_markers]

### `torch.nn.Module.__call__` hook misses `.forward()` direct calls
**Symptom:** Hook captures `SiglipVisionModel` (1152-dim hidden) as
"policy output", derives `action_dim=1152` instead of 7. Top-level
policy never fires the hook because `.select_action` calls
`self.forward(batch)` directly, bypassing `__call__`.
**Fix:** Can't be solved by a better `__call__` hook alone. Fall back
to `infer_io_spec`'s 3-source merge (README + config.json + probe
partial) instead of trusting the hook's captured shape. Never trust a
captured `action_dim` that matches a known vision-encoder hidden dim
(1152 Siglip, 1280 DINOv2, etc.). [mem:feedback_hook_captures_submodule]

### torch.compile replaces `.forward` AFTER our hooks install
**Symptom:** Targeting `PI0Pytorch,PI05Pytorch,SmolVLA` explicitly, still
only `SiglipVisionModel` is captured. Confirmed on job 113814.
**Fix:** torch.compile wraps `.forward` with a compiled graph on first
call, after `__init__` → class-forward patch has already run. Don't
fight the compile — use the `infer_io_spec` merge path, which reads
config.json shape fields directly. Document that probe_run cannot shape-
capture compiled policies; callers should route to infer_io_spec when
the policy is known to be compiled (pi0, pi0.5, smolvla via lerobot).
[mem:feedback_probe_torch_compile]

---

## Rule of thumb

If all three traps above would apply to a policy (lerobot VLM → compiled
forward, direct `.forward` calls, pre-forward "Warming up" log), do not
use probe_run for io_spec. Use `extract_eval_protocol` + `infer_io_spec`
(config.json path) instead.
