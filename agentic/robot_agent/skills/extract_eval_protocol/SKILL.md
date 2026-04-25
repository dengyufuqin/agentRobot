---
name: extract_eval_protocol
description: "Pull the exact eval rules for a (model, benchmark, checkpoint) tuple from HF model card + config files + upstream paper + author eval repo. Emit eval_protocol.json with source citations. Gates run_benchmark — missing any field produces large alignment gaps (e.g. PullCube 40% vs paper 92%). Every time we've assumed a default, it was wrong."
version: 0.1.0
category: analysis
parameters:
  policy:
    type: string
    description: "Policy name (e.g. openvla, octo, pi0.5)"
    required: true
  benchmark:
    type: string
    description: "Benchmark id (e.g. maniskill, libero_spatial, robocasa, robotwin)"
    required: true
  checkpoint:
    type: string
    description: "HF id or local path to checkpoint directory. Local path is preferred — the skill reads README.md + config.json + dataset_statistics.json + preprocessor_config.json."
    required: true
  eval_repo:
    type: string
    description: "Git URL or local path to the author's eval code (e.g. https://github.com/RobotControlStack/vlagents). Often linked from the HF README — the skill follows the link."
    required: false
  paper_url:
    type: string
    description: "arxiv or project-page URL of the upstream paper. Used to cite per-task max_episode_steps, image_resolution, etc."
    required: false
  out:
    type: string
    description: "Where to write eval_protocol.json. Defaults to $AGENTROBOT_ROOT/agentic/robot_agent/eval_protocols/<policy>_<benchmark>.json"
    required: false
  validate:
    type: string
    description: "If 'true', exit nonzero when any field is still [MANUAL] or missing. run_benchmark calls this as a gate."
    default: "false"
requires:
  bins: [python3]
timeout: 60
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/extract_eval_protocol/extract_eval_protocol.py \
    --policy "{policy}" \
    --benchmark "{benchmark}" \
    --checkpoint "{checkpoint}" \
    $([ -n "{eval_repo}" ] && echo "--eval-repo {eval_repo}") \
    $([ -n "{paper_url}" ] && echo "--paper-url {paper_url}") \
    $([ -n "{out}" ] && echo "--out {out}") \
    $([ "{validate}" = "true" ] && echo "--validate")
---

# Extract Eval Protocol

Before submitting any SLURM benchmark job, this skill pulls the exact rules the
upstream authors used for eval. It is a **gate**: run_benchmark refuses to
submit if `eval_protocol.json` for the (policy, benchmark, checkpoint) tuple
doesn't exist, or still has `[MANUAL]` placeholders.

## Why

Every time we assumed an eval default, we were wrong. Concrete losses:
- OpenVLA × ManiSkill PullCube: **40% vs paper 92%** because we missed
  `max_episode_steps=300` (env default 50), `img_res=256`, and — today's
  discovery — `center_crop=False` + RPD does no 256→224 resize.
- OpenVLA × LIBERO: needed 180° image flip, pulled from observational testing
  not spec.
- Octo × ManiSkill: needed `HumanCameraWrapper` overwriting base_camera with
  `env.render()`, not documented anywhere except in `juelg/agents`
  (since renamed to `RobotControlStack/vlagents`).

## Protocol schema (eval_protocol.json)

```json
{
  "policy":       "openvla",
  "benchmark":    "maniskill",
  "checkpoint":   "Juelg/openvla-7b-finetuned-maniskill",
  "fields": {
    "image_resolution":         { "value": 256,  "source": "paper §IV.A: image resolution 256×256" },
    "max_episode_steps":        { "value": 300,  "source": "paper §IV.A: increased max_episode_length to 300" },
    "camera":                   { "value": "HumanCameraWrapper — base_camera replaced by env.render()", "source": "vlagents/wrappers.py:22-29" },
    "control_mode":             { "value": "pd_ee_delta_pose", "source": "vlagents/ppo_rgb_rpd.py:371" },
    "obs_mode":                 { "value": "rgb",  "source": "vlagents/ppo_rgb_rpd.py:370" },
    "prompt_format":            { "value": "In: What action should the robot take to {instruction.lower()}?\\nOut:", "source": "vlagents/policies.py:421" },
    "image_flip_180deg":        { "value": false, "source": "vlagents has no flip; human_camera is already upright" },
    "resize_method":            { "value": "lanczos",  "source": "vlagents/policies.py:429 comment 'use lanczos'" },
    "image_resize_before_model": { "value": false, "source": "vlagents sends 256×256 directly; HF processor resizes internally" },
    "center_crop":              { "value": false, "source": "vlagents/policies.py:435 processor() no center_crop kwarg" },
    "gripper_post_processing":  { "value": "a[-1] = a[-1] * 2 - 1.0  # rescale [0,1]→[-1,1] only, NO binarize, NO invert", "source": "vlagents/evaluator_envs.py:172" },
    "wait_steps":               { "value": 0,    "source": "vlagents/evaluator_envs.py: no warmup loop (our eval uses 10 for init-pose alignment)" },
    "unnorm_key":               { "value": "maniskill_human:7.0.0",  "source": "Juelg/openvla-7b-finetuned-maniskill/dataset_statistics.json key" },
    "human_render_camera_configs": { "value": {"width": 256, "height": 256}, "source": "vlagents/ppo_rgb_rpd.py:378" },
    "num_eval_envs":            { "value": 16,   "source": "vlagents/ppo_rgb_rpd.py --num_eval_envs default" },
    "sim_backend":              { "value": "gpu", "source": "vlagents/ppo_rgb_rpd.py:366" }
  }
}
```

Each field has `value` + `source`. Fields left as `"[MANUAL]"` or missing
cause `--validate` to exit nonzero.

## Workflow

1. **Checkpoint dir scan** — read README.md, config.json, dataset_statistics.json,
   preprocessor_config.json. Extract `unnorm_key`, default prompts, HF eval-repo
   links.
2. **Paper extraction** — if `--paper-url` given, fetch PDF and grep for
   `max_episode_length`, `image resolution`, `control_mode`, `episode length`.
3. **Eval-repo extraction** — if `--eval-repo` given (or auto-detected from
   README), clone shallow and grep for `env_kwargs`, `control_mode`, `obs_mode`,
   `render_mode`, `center_crop`, `flip`, `resize`, `gripper`.
4. **Emit JSON** — write to `$AGENTROBOT_ROOT/agentic/robot_agent/eval_protocols/<policy>_<benchmark>.json`.
5. **Validate mode** — `--validate` returns 0 only if every field has a concrete
   value AND a non-empty source. Called by `run_benchmark` before SLURM submission.

## Integration with run_benchmark

`run_benchmark` first calls this skill with `--validate`. If it fails, it
prints a pointer:

```
❌ eval_protocol.json missing fields:
   center_crop (source required)
   gripper_post_processing (source required)
Run: /skill extract_eval_protocol --policy openvla --benchmark maniskill ...
```

and refuses to submit. This forces the protocol extraction to happen
BEFORE compute is wasted.
