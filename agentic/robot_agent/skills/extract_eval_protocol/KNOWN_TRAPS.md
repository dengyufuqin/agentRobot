# extract_eval_protocol — known traps

Per-skill harness doc. `extract_eval_protocol` produces the
`eval_protocols/<policy>_<benchmark>.json` file that
`run_benchmark.py::run_protocol_gate` validates before submitting a
SLURM job. Every trap below is a field we learned to extract only
after a 0% run.

Back-references (`[mem:…]`) point at the full memory entry.

---

### Protocol is extracted from paper + README + author eval repo — not guessed
**Symptom:** Missing any one of max_episode_steps / img_res / camera /
prompt / control_mode / flip / center_crop produces a large alignment
gap. We lost two full SLURM cycles on OpenVLA × ManiSkill PullCube
before we learned to do the upstream-rule extraction first.
**Fix:** Before running a new combo, read the paper, the model card,
and the authors' eval code (follow the GitHub link on the HF card).
Capture every field the eval code branches on. Do NOT fill defaults
from our policy_server. [mem:feedback_extract_eval_protocol_first]

### Shape fields gate compatibility, not just cosmetic
**Symptom:** pi0fast trained dual-arm (state_dim=16) fed a RoboCasa
client sending 8D → `shape (8,) vs (16,)` at first inference. The old
protocol didn't record state_dim, so the gate couldn't catch it.
**Fix:** Always extract `state_dim`, `action_dim`, `required_obs_keys`,
`obs_wrapper` from `config.json`. These are the four fields that gate
"will this even run?" separately from "does it produce good actions?".
[mem:feedback_extract_eval_protocol_shape_fields]

---

## Current schema

The protocol JSON must include:
- `image_resolution` (int or [H, W])
- `max_episode_steps` (int)
- `camera` (e.g. "agentview", "base_camera", "render_camera")
- `flip` (bool — 180° rotation, LIBERO/RoboCasa-style)
- `center_crop` (bool)
- `unnorm_key` (str)
- `state_dim` (int)
- `action_dim` (int)
- `required_obs_keys` (list)
- `obs_wrapper` (str | null — e.g. "HumanCameraWrapper" for RPD ManiSkill)

Missing any field that the downstream `run_benchmark` --flag injection
depends on causes the gate to fail closed.
