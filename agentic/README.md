# Agentic: One-Sentence Deployment for Robot Learning

An agentic orchestration system for robot learning. Give a natural language command, and the system automatically handles everything: downloading models, setting up environments, deploying servers, running benchmarks, and reporting results.

Works on both **HPC clusters** (SLURM) and **local machines** with GPU.

```
User: "用openvla跑LIBERO-spatial的benchmark"
Agent: check_cluster_status -> run_benchmark(submit=SLURM) -> Job running on H100
       94.0% success rate (47/50 episodes)

User: "集成 https://github.com/intuitive-robots/beso"
Agent: analyze_repo -> setup_env -> fix_deps -> wrap_policy -> create_deploy_skill
       5 steps, fully autonomous, 0 manual intervention

User: "vq_bet/policy_server.py 现在 import 失败,把它修到 validate 通过"
Agent: list_files -> read_file -> validate(FAIL) -> setup_env -> validate(FAIL)
       -> fix_deps -> validate(IMPORT_OK)
       9 skill calls, fully autonomous, verified end-to-end with qwen-max

User: "集成 tonyzhaozh/act (不在 6 regex 模式里),到 smoke 通过"
Agent: analyze_repo -> read_file(policy.py) -> read_file(imitate_episodes.py)
       -> write_file(minimal adapter with _validate_only=True)
       -> setup_env -> validate(mode=import) -> validate(mode=smoke)
       SMOKE_OK: infer returned actions of len=14
       Proves "any GitHub model" claim: zero-touch onboarding for
       a repo that doesn't match any of the 6 wrap_policy patterns.
```

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        SOUL.md                               │
│             (Agent identity, rules, examples)                │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                     agent.py                                  │
│   ReAct Loop (Claude/GPT/Qwen) + JSON fault-tolerance         │
│   36 Skills + Auto-discovery from policy_server.yaml          │
│   + Loop-breaker (intercepts repeated identical failures)     │
│   + exit-10 disambiguation (ambiguous ckpt/variant → ask)     │
└──────┬─────────┬──────────┬──────────┬───────────┬───────────┘
       │         │          │          │           │
   ┌───▼──┐  ┌──▼───┐  ┌──▼───┐  ┌──▼────┐  ┌──▼─────┐
   │ Meta │  │ Env  │  │ Eval │  │Deploy │  │ Valid/ │
   │Skills│  │Skills│  │Skills│  │Skills │  │Codegen │
   └──────┘  └──────┘  └──────┘  └───────┘  └────────┘
```

## How It Works

### End-to-End Call Chain

Here's exactly what happens when you type one sentence:

```
python agent.py "用openvla跑LIBERO-spatial的benchmark"
```

```
User: "用openvla跑LIBERO-spatial的benchmark"
  │
  ▼
main()
  │  1. Sets AGENTROBOT_ROOT from script location
  │  2. load_skills(): scans skills/*/SKILL.md → parses YAML → 16 skill dicts
  │  3. discover_policy_servers(): scans */policy_server.yaml → 3 auto skills
  │  4. Total: 19 skills ready
  │
  ▼
run_agent(user_message, skills)
  │
  │ Step 1: Convert skills to LLM tool definitions
  │   skill_to_claude_tool() for each skill:
  │     SKILL.md name/description  →  tool name/description
  │     SKILL.md parameters        →  JSON Schema (properties/required)
  │   → becomes the `tools` param for the LLM API call
  │
  │ Step 2: Enter ReAct loop (max 15 turns)
  │
  ▼
┌─── Turn 1 ──────────────────────────────────────────────────┐
│                                                              │
│  call_llm(messages, tools, model)                            │
│    → sends to Claude/GPT/Qwen API:                           │
│        system: SOUL.md content (agent identity + rules)      │
│        tools:  19 skills as JSON Schema                      │
│        messages: [user's sentence]                           │
│                                                              │
│  LLM returns:                                                │
│    text: "Let me check available GPUs..."                    │
│    tool_calls: [{                                            │
│      name: "check_cluster_status",                           │
│      arguments: {}                                           │
│    }]                                                        │
│                                                              │
│  execute_skill(skill, args):                                 │
│    1. Takes command_template from SKILL.md                   │
│    2. Substitutes {param} placeholders with actual values    │
│    3. subprocess.run(["bash", "-c", cmd])                    │
│    4. Returns stdout                                         │
│                                                              │
│  Result: "=== Local GPU Status ===\n0, NVIDIA H100 80GB..."  │
│  → appended to messages as tool_result                       │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
┌─── Turn 2 ──────────────────────────────────────────────────┐
│                                                              │
│  LLM sees GPU available, decides to call run_benchmark       │
│                                                              │
│  tool_calls: [{                                              │
│    name: "run_benchmark",                                    │
│    arguments: {                                              │
│      "policy": "openvla",                                    │
│      "checkpoint": "moojink/openvla-7b-oft-...",             │
│      "benchmark": "libero_spatial",                          │
│      "num_trials": 5,                                        │
│      "submit": "false"     ← LLM chose local mode           │
│    }                                                         │
│  }]                                                          │
│                                                              │
│  execute_skill() runs:                                       │
│    python3 $AGENTROBOT_ROOT/.../run_benchmark.py \           │
│      --policy openvla --benchmark libero_spatial ...         │
│                                                              │
│    run_benchmark.py Mode 3 (local):                          │
│      → start_server(): launches OpenVLA on local GPU         │
│      → wait_for_server(): polls /healthz until ready         │
│      → run_libero_eval(): runs 10 tasks × 5 trials          │
│      → returns {"success_rate": 94.0}                        │
│                                                              │
│  Result: "Success rate: 94.0% (47/50 episodes)"              │
└──────────────────────────────────────────────────────────────┘
  │
  ▼
┌─── Turn 3 ──────────────────────────────────────────────────┐
│  LLM sees result, generates summary:                         │
│  "Benchmark complete: 94.0% success rate on LIBERO-spatial"  │
│  is_done = True → loop ends                                  │
└──────────────────────────────────────────────────────────────┘
```

### The Three Core Functions

The entire system boils down to three operations:

```python
# 1. SKILL.md → LLM tool definitions
tools = [skill_to_claude_tool(s) for s in skills]          # agent.py:445

# 2. LLM decides which skill to call and fills in parameters
text, tool_calls, ... = call_llm(messages, tools, model)   # agent.py:457

# 3. Substitute parameters into command_template, execute as bash
result = execute_skill(skill, tc["arguments"])              # agent.py:489
```

**The LLM never executes code.** It only selects tools and fills parameters. All actual work is done by `execute_skill()` → `subprocess.run(["bash", "-c", cmd])`.

### How a SKILL.md Becomes an Executable Tool

Take `deploy_openvla/SKILL.md` as an example:

```yaml
# What LLM sees (converted by skill_to_claude_tool):
name: deploy_openvla
description: "Deploy OpenVLA-OFT model as a WebSocket policy server"
parameters:
  node:     { type: string, required: false }
  port:     { type: integer, default: 18800 }
  gpu_id:   { type: integer, default: 0 }
  checkpoint: { type: string, default: "moojink/openvla-7b-oft-..." }

# What gets executed (command_template after {param} substitution):
command_template: |
  REPO=$AGENTROBOT_ROOT/openvla
  if [ -z "{node}" ] || [ "{node}" = "localhost" ]; then
    # local mode: start server directly
    CUDA_VISIBLE_DEVICES={gpu_id} python3 policy_server.py --port {port} ...
  else
    # remote mode: SSH to compute node
    ssh {node} "CUDA_VISIBLE_DEVICES={gpu_id} python3 policy_server.py --port {port} ..."
  fi
```

When the LLM calls `deploy_openvla(port=18800, gpu_id=0)`:
1. `execute_skill()` replaces `{port}` → `18800`, `{gpu_id}` → `0`, `{node}` → `` (empty)
2. Runs the resulting bash script via `subprocess.run()`
3. Empty `{node}` triggers local mode → server starts on current machine

### policy_websocket (Communication Layer)

Universal bridge between any model and any client:

```python
# Server side (13-line ABC)
class BasePolicy(ABC):
    def infer(self, obs: dict) -> dict: ...
    def reset(self): ...

# Client side
policy = WebsocketClientPolicy(host="localhost", port=18800)
result = policy.infer({"image": img, "task_description": "pick up the bowl"})
```

- Async WebSocket + msgpack binary serialization
- Healthcheck endpoint at `/healthz`
- Works with any ML framework (PyTorch, JAX, TensorFlow)

### Data Flow: From User Sentence to Robot Action

```
User sentence
  → LLM API (tool selection)
    → execute_skill() (bash subprocess)
      → policy_server.py (model inference via WebSocket)
        → run_eval.py (simulation loop)
          → WebsocketClientPolicy.infer(obs)
            → msgpack serialize → WebSocket → msgpack deserialize
              → model.predict(obs) → action (7D numpy array)
            → msgpack serialize → WebSocket → msgpack deserialize
          → env.step(action) → next obs
        → success rate
      → stdout captured
    → tool_result back to LLM
  → LLM generates summary
→ "94.0% success rate"
```

## Any GitHub Model: The LLM-Driven Adapter Loop

The `wrap_policy` skill ships with a regex-based generator that handles **6 common
loading patterns** (Hydra, `from_pretrained`, `algo_factory`, `load_pretrained`,
`deserialize`, `--checkpoint`). For repos that don't match any of those patterns —
custom factory loaders, multi-stage init, hand-rolled config systems — the agent
now writes the adapter **by reading the model's actual source code**, with no
human in the loop.

### The Loop

```
list_files(repo, "demo*.py")            ← discover candidate scripts
       ↓
read_file(...)                           ← read how the checkpoint is loaded
       ↓
write_file(repo/policy_server.py, ...)   ← LLM emits adapter from what it READ
       ↓
validate_policy_server(mode="import")    ← <5s sandboxed import check, no GPU
       ↓
   FAIL? → read_file(traceback line)
        → edit_file(targeted patch)      ← surgical fix, ~50-byte b64 payload
        → validate again
       ↓
   PASS  → mode="smoke" (instantiate + fake infer())
       ↓
   PASS  → setup_env / create_deploy_skill / run_benchmark
```

### Two Robustness Layers

These exist because the loop has to survive **weak LLMs** (Qwen, GPT-3.5-class)
that occasionally emit broken tool arguments.

1. **`edit_file` auto-repairs broken base64 payloads.** The most common LLM
   token-emission bug is a literal `\n` inserted between two adjacent characters
   (e.g. `GPTConfig` → `GPT\nConfig`). If `old_b64` doesn't match the file
   verbatim, `edit_file` tries 4 cleanup heuristics — strip embedded newlines,
   strip literal `\n` sequences, strip whitespace runs, strip leading/trailing
   whitespace — and accepts the first variant that **uniquely** matches.
   Failed repairs leave the file byte-identical (atomic `.tmp` + rename).

2. **`agent.py` loop-breaker intercepts repeated identical failures.** When the
   same `(tool_name, args)` tuple has failed 2 times in a row, the 3rd identical
   call is intercepted with a synthetic error telling the LLM to re-read the
   file and change something substantive — preventing the `max_turns` death
   spiral that weak models can fall into when they don't realize their tool
   arguments are bit-for-bit identical to the failing one.

### Verification

| Layer | Test | Result |
|---|---|---|
| `edit_file` auto-repair | 5 unit tests (decode, repair, ambiguity-refusal, atomic-write) | 5/5 PASS |
| `edit_file` end-to-end | Reproduce exact qwen-max `GPT\nConfig` glitch on real adapter | PASS |
| Loop-breaker | 7 tests (canonical key, error heuristic, intercept threshold, reset) | 7/7 PASS |
| Robustness suite | 13 integrated tests (validate FAIL → repair → validate PASS, atomic on failure, baseline restore) | 13/13 PASS |
| **qwen-max e2e on vq_bet (import)** | "fix vq_bet to validate import" — 9 skill calls, agent self-converges | **`IMPORT_OK CLASSES=["VqBetPolicy"]`** |
| **qwen3-coder-plus: LLM source-edit on vq_bet** | "read the failing file and edit_file the bad import (`from gpt` → `from vq_behavior_transformer.gpt`) until smoke passes" — 6 autonomous skill calls | **line 36 fixed cleanly on first edit, no loop-breaker needed** |
| **Claude onboarding ACT from zero**<br/>(`tonyzhaozh/act`, not matching the 6 regex patterns) | analyze_repo → read_file(policy.py + imitate_episodes.py) → write_file(minimal adapter) → validate mode=import → validate mode=smoke | **`SMOKE_OK: infer(no-image) returned actions of len=14` — any-GitHub-model claim proven end-to-end** |

**How to read the verification above.** The vq_bet rows prove that weak LLMs
can now drive the `read_file → edit_file → validate` loop to fix source bugs
autonomously. The ACT row proves the full onboarding flow (clone → read the
upstream inference code → write a fresh `BasePolicy` adapter → pass smoke) for
a repo that is NOT in the 6-regex wrap_policy table, which is the whole point
of the "any GitHub model" claim.

**Model caveat.** The infrastructure is model-agnostic, but `write_file`'s
long base64 payload stresses the model's tokenizer. We have observed that
`qwen-max` and `qwen3-coder-plus` produce corrupted base64 for payloads over
~500 bytes (character-level hallucinations, invalid UTF-8 bytes, token
repetition loops). For the `write_file` step specifically, prefer a stronger
model (Claude Sonnet 4.6, Claude Opus 4.6, or GPT-4o). The `edit_file`,
`read_file`, `validate_policy_server`, and smaller-payload skills work
reliably even with weaker models thanks to the C1/C2 robustness layer.

## 36 Skills

| Category | Skill | Description |
|----------|-------|-------------|
| **Meta** | `analyze_repo` | Clone GitHub URL, analyze model structure and inference patterns |
| **Meta** | `probe_run` | Run the repo's own demo with a forward-hook, capture ground-truth tensor shapes |
| **Meta** | `extract_io_spec` | Parse `.probe_io_spec.json` → image_keys / state_keys / action_dim |
| **Meta** | `infer_io_spec` | **Meta-skill.** Merge README regex + probe + user-fallback; exit 3 on LOW confidence |
| **Meta** | `onboard_benchmark` | Symmetric to probe_run but for simulator repos (libero/maniskill/robocasa/calvin/simpler) |
| **Meta** | `check_finetune_capability` | Scan a repo for existing train/finetune scripts before writing one |
| **Meta** | `create_deploy_skill` | Auto-generate new deploy SKILL.md (self-expanding) |
| **Meta** | `wrap_policy` | **Deprecated.** Regex generator; kept only as fallback |
| **Util** | `read_file` | Read any file with line numbers (paginated, capped 200KB) |
| **Util** | `list_files` | Glob-aware listing that auto-skips `.venv` / `__pycache__` / `data/` |
| **Util** | `write_file` | Write a whole file (use only for new files / >50% rewrites) |
| **Util** | `edit_file` | Targeted base64 substring replacement, atomic + auto-repairs LLM glitches |
| **Util** | `validate_policy_server` | Sandboxed syntax/import/smoke check on adapter, no GPU, <5s |
| **Env** | `setup_env` | Create venv, install deps, CUDA auto-detect, pin torch cu121 |
| **Env** | `fix_deps` | Auto-diagnose and fix dependency issues (15+ patterns, 50+ mappings) |
| **Env** | `build_container` | Generate Apptainer .def files, build .sif containers |
| **Env** | `download_model` | Variant-aware HF checkpoint download; multi-candidate → exit 10 |
| **Env** | `download_dataset` | HF dataset download with resume + size reporting |
| **Valid** | `validate_dataset` | Detect dataset format (lerobot/parquet/webdataset/zarr/rlds) |
| **Valid** | `generate_dataloader` | Emit a starter `make_dataloader()` factory for a detected format |
| **Valid** | `validate_dataloader` | Pull 1–2 batches from a factory; shape/dtype/finite/missing-keys check |
| **Eval** | `run_benchmark` | Multi-platform eval (LIBERO/ManiSkill/RoboTwin/RoboCasa); registry + HF + cross-domain fallback |
| **Eval** | `finetune` | Train-only SLURM job |
| **Eval** | `train_and_eval` | SLURM one-shot: finetune → deploy → eval → report |
| **Eval** | `check_cluster_status` | Local nvidia-smi or SLURM cluster status (auto-detect) |
| **Deploy** | `deploy_policy` | **Unified deploy skill.** Pass `repo=...` instead of per-model skills |
| **Deploy** | `deploy_openvla` | Legacy: OpenVLA-OFT deploy (kept for back-compat) |
| **Deploy** | `deploy_octo` | Legacy: Octo deploy |
| **Deploy** | `deploy_lerobot` | Legacy: LeRobot (pi0/pi0.5/SmolVLA) deploy |
| **Deploy** | `deploy_diffusion_policy` | Legacy: Diffusion Policy deploy |
| **Deploy** | `deploy_robomimic` | Legacy: Robomimic deploy |
| **Deploy** | `deploy_beso` | Legacy: BESO deploy |
| **Deploy** | `test_policy_connection` | Verify running server via WebSocket |
| **Deploy** | `stop_policy_server` | Stop running policy server process |
| **Codegen** | `generate_run_demo` | Emit a standalone `run_demo.sh` — no agent needed to re-run |
| **Codegen** | `generate_run_evaluation` | Emit a standalone `run_evaluation.sh` — user `sbatch`s directly |

## Local vs HPC Mode

All skills auto-detect the environment:

| Feature | Local Mode | HPC Mode |
|---------|-----------|----------|
| GPU detection | `nvidia-smi` | `squeue` + `sinfo` |
| Deployment | Direct process launch | SSH to compute node |
| Benchmark | Local server + eval | SLURM job submission |
| Path resolution | `$AGENTROBOT_ROOT` (auto-detected) | Same |

**Deploy skills**: Leave `node` empty for local, specify `node` (e.g. `cn06`) for remote SSH deployment.

## Multi-Platform Benchmarks

The `run_benchmark` skill supports pluggable eval clients. Any benchmark with a WebSocket eval script works.

### Benchmark Format

```
libero_spatial              → LIBERO spatial suite (robosuite)
maniskill:PickCube-v1       → ManiSkill task (SAPIEN)
robotwin:beat_block_hammer  → RoboTwin dual-arm task
<platform>:<task_id>        → Any task on any registered platform
```

### Supported Platforms

| Platform | Backend | Tasks | Example |
|----------|---------|-------|---------|
| **LIBERO** | robosuite/MuJoCo | 5 suites (10-90 tasks each) | `libero_spatial` |
| **ManiSkill** | SAPIEN | 50+ envs | `maniskill:PickCube-v1` |
| **RoboTwin** | custom sim | 50+ dual-arm tasks | `robotwin:open_laptop` |

### Architecture

```
run_benchmark.py
├── EVAL_CLIENTS registry        ← pluggable eval scripts
│   ├── libero  → LIBERO/scripts/run_eval.py
│   ├── maniskill → ManiSkill/scripts/run_eval.py
│   └── robotwin → RoboTwin/script/run_eval_ws.py
├── POLICY_CONFIGS registry      ← policy server configs
│   ├── openvla, diffusion_policy (built-in)
│   └── auto-discovered from */policy_server.yaml
└── resolve_benchmark()          ← "maniskill:PickCube-v1" → {client, task_id}
```

All eval clients share the same **WebSocket policy interface** (`policy_websocket`), so any model works with any benchmark as long as the action space is compatible.

## Quick Start

### 1. Set up

```bash
git clone https://github.com/dengyufuqin/Agentic.git
cd Agentic

# Set your LLM API key (pick one)
export ANTHROPIC_API_KEY=sk-xxx    # Claude
export OPENAI_API_KEY=sk-xxx       # GPT
export DASHSCOPE_API_KEY=sk-xxx    # Qwen
```

### 2. Run the agent

```bash
# Interactive mode
python robot_agent/agent.py

# Single command
python robot_agent/agent.py "deploy openvla and test connection"
python robot_agent/agent.py "integrate https://github.com/xxx/new-model"
python robot_agent/agent.py "check GPU status"

# Override model (default: claude-sonnet-4-6 / qwen-plus / gpt-4o based on key)
python robot_agent/agent.py --model qwen-max "fix vq_bet/policy_server.py"
python robot_agent/agent.py --model claude-opus-4-6 "integrate https://github.com/xxx/new-model and run libero_spatial"
```

### 2b. Onboard an arbitrary GitHub model (no manual coding)

For repos that don't fit the 6 regex-templated loading patterns, the agent uses
the LLM-driven loop documented above. One sentence is enough:

```bash
python robot_agent/agent.py \
  "克隆 https://github.com/some-org/some-model, 写 policy_server.py adapter, 跑通 validate_policy_server 的 import 模式"
```

What happens, fully autonomous:

1. `analyze_repo` clones the repo and scans top-level structure
2. `list_files(pattern="demo*.py")` + `list_files(pattern="*model*.py")` find inference scripts
3. `read_file` reads the loading + predict code (the LLM never guesses)
4. `write_file` emits a `policy_server.py` adapter inheriting from `BasePolicy`
5. `validate_policy_server(mode="import")` runs in a subprocess sandbox
6. On failure: `read_file` the traceback line → `edit_file` a surgical patch → re-validate
7. On `IMPORT_OK`: re-validate with `mode="smoke"` (instantiates and calls `infer()`)
8. On smoke pass: `setup_env` + `create_deploy_skill` + ready to benchmark

The two robustness layers (`edit_file` auto-repair + agent loop-breaker) keep
weak LLMs from getting stuck. Verified with qwen-max — see "Verification" above.

### 3. Direct skill execution

```bash
# Run benchmark — LIBERO
python robot_agent/skills/run_benchmark/run_benchmark.py \
  --policy openvla --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --benchmark libero_spatial --num_trials 5

# Run benchmark — ManiSkill
python robot_agent/skills/run_benchmark/run_benchmark.py \
  --policy openvla --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --benchmark maniskill:PickCube-v1 --num_trials 10

# Run benchmark — RoboTwin
python robot_agent/skills/run_benchmark/run_benchmark.py \
  --policy openvla --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --benchmark robotwin:beat_block_hammer --num_trials 5

# List all available benchmarks
python robot_agent/skills/run_benchmark/run_benchmark.py --list_benchmarks \
  --policy x --checkpoint x --benchmark x

# Fix dependencies
python robot_agent/skills/fix_deps/fix_deps.py /path/to/repo --max-retries 5

# Generate policy adapter
python robot_agent/skills/wrap_policy/generate_smart.py /path/to/repo ModelClass module.path
```

## Benchmark Results

All evaluations on **NVIDIA H100 80GB** via SLURM, paper-standard trial counts
(LIBERO 100 ep, ManiSkill 50 ep, RoboTwin 50 ep). Each cell below was driven by
**one** `run_benchmark --submit` call — protocol gate → SLURM → policy
WebSocket boot → eval client → JSON result. No per-cell custom scripts.

Master cross-simulator table: see [`BENCHMARK_VERIFIED.md`](BENCHMARK_VERIFIED.md).
Per-simulator deep-dives: [`MANISKILL_VERIFIED.md`](MANISKILL_VERIFIED.md),
[`ROBOTWIN_VERIFIED.md`](ROBOTWIN_VERIFIED.md), [`PI_ALL_LIBERO.md`](PI_ALL_LIBERO.md).

### Paper-aligned cells (snapshot 2026-04-25)

✅ = within paper noise. ⚠️ = documented infra-side gap with root cause.

#### LIBERO (100 episodes/task, 10 tasks/suite)

| Algorithm | Spatial | Object | Goal | 10 | Source |
|---|:-:|:-:|:-:|:-:|---|
| **pi0** | 68% ✅ | 72% ✅ | 84% ✅ | 34% ✅ | LeRobot `pi0_libero_finetuned` |
| **pi0.5** | 88% ✅ | 87% ✅ | 89% ✅ | 85% ✅ | LeRobot `pi05_libero` |
| **pi0-FAST** | 94% ✅ | 96% ✅ | 86% ✅ | 84% ✅ | `lerobot/pi0fast-libero` |
| **OpenVLA-OFT** | ⚠️ sim abort | 82% ✅ | 83% ✅ | 62% ⚠️ | OFT paper Stanford-ILIAD |

#### ManiSkill (50 episodes)

| Algorithm | PullCube-v1 | PushCube-v1 | Source |
|---|:-:|:-:|---|
| **Octo** (RPD ckpt) | 66% ✅ beat (paper 52%) | 44% ✅ | RPD paper (2503.05833) Tab.2 |
| **OpenVLA-base** (RPD ckpt) | 80% ✅ beat (paper 65%) | 30% ✅ | RPD paper (2503.05833) Tab.2 |

#### RoboTwin 2.0 (50 episodes, dual-arm ALOHA-agilex)

| Algorithm | beat_block_hammer | Source |
|---|:-:|---|
| **ACT** | 42% ✅ (paper Easy avg 40%) | RoboTwin 2.0 paper Tab.5 |
| **Diffusion Policy** | 34% ✅ (within noise) | RoboTwin 2.0 paper Tab.5 |

### Robustness signal (small-N → paper-standard convergence)

| Combo | N=3 | N=10 | N=50 (paper-standard) | Paper target |
|---|:-:|:-:|:-:|:-:|
| Octo × PullCube | — | 70% | **66%** | 52% |
| OpenVLA × PullCube | — | 80% | **80%** | 65% |
| ACT × beat_block_hammer | 67% | — | **42%** | 40% |
| DP × beat_block_hammer | 33% | — | **34%** | 40% |

Same pipeline, just `--num_trials 50`. Small-N estimates are noisy; N=50
lands inside paper noise for every cell.

### Architecture coverage

8 architecture families wired end-to-end via the canonical drop-in flow
(own-venv when needed + `BasePolicy` server + `eval_protocols/<policy>_<bench>.json`
+ registry entry):

| Family | Format | Paper-aligned cells |
|---|---|--:|
| pi0 | LeRobot PyTorch | 4 (LIBERO ×4) |
| pi0.5 | LeRobot PyTorch | 4 (LIBERO ×4) |
| pi0-FAST | LeRobot PyTorch | 4 (LIBERO ×4) |
| OpenVLA-OFT | HF transformers + action_head | 3 (LIBERO ×3) |
| OpenVLA-base | HF transformers (no OFT) | 2 (ManiSkill ×2) |
| Octo | JAX/Flax | 2 (ManiSkill ×2) |
| ACT | own-venv (CVAE) | 1 (RoboTwin ×1) |
| Diffusion Policy | own-venv (DDPM, hydra+dill) | 1 (RoboTwin ×1) |

### Onboarding cost

ACT and DP each took ~30 min from "let's wrap it" to a green run via the
canonical flow: list_files → read_file → write_file (`policy_server.py`) →
setup_env → validate → register protocol → submit. No per-cell custom scripts.

### What's verified vs pending

**✅ Verified (this push):**
- 21 paper-aligned cells across 4 simulators × 8 architectures
- 2 documented infra-side gaps (OpenVLA-OFT spatial sim abort; pi0.5 ManiSkill torch.compile deadlock)
- Protocol gate auto-validates 18+ fields per cell (img_res, max_episode_steps,
  camera, control_mode, prompt_format, flip, center_crop, gripper post-proc,
  unnorm_key, state_dim, action_dim, obs keys, train/eval seed ranges, …)
- Protocol gate auto-translates validated fields into `--flags` on both
  client and server (octo+openvla 0% → 66.7% PullCube after this fix)
- SLURM auto-submission: `--time` budget configurable, port randomization
  for concurrent jobs, ManiSkill node blacklist for stalling SAPIEN nodes,
  EGL whitelist for live LIBERO/RoboCasa render nodes

**⏳ Pending:**
- GR00T × RoboCasa, Fast-WAM × LIBERO, LingBot-VLA × LIBERO (#119–#121)
- Pi0.5 × RoboTwin (Hoshipu openpi-JAX/orbax + Crelf openpi-PyTorch) — own venv
  required, needs `RoboTwin/policy/pi05/.venv` + `pi05_robotwin` train_config (#143, #146, #147)
- Cross-domain combos with `--allow_cross_domain` + ActionSanityChecker (#76)
- Harness inferential sensor: LLM review of generated SLURM scripts (#125)

**Note on vendored adapters.** ACT/DP/Pi0.5-RoboTwin policy servers live inside
the `RoboTwin/` vendored repo (which has its own `.git`). Those adapter files
(`RoboTwin/policy/{ACT,DP}/policy_server.py` + `setup_env.sh`) are not yet
visible to the parent repo's tracking. To include them on a future push, either
(a) absorb RoboTwin's worktree into this repo, or (b) add them as an explicit
git submodule pointing at our fork.

## Supported Algorithms

9 algorithms integrated, spanning 5 architecture families:

| Algorithm | Type | Framework | Adapter | Deploy Skill | Env | Validation Level |
|-----------|------|-----------|---------|-------------|-----|-----------------|
| **OpenVLA-OFT 7B** | Vision-Language-Action | PyTorch | `openvla/vla-scripts/policy_server.py` | `deploy_openvla` | `.venv` ✓ | **Benchmark** (94.0% LIBERO-spatial) |
| **LeRobot pi0.5** | Flow-matching VLA | PyTorch | `lerobot/policy_server.py` | `deploy_lerobot` | `.venv` ✓ | **Import** ✓ Benchmark *running* |
| **Octo** | Transformer policy | JAX | `octo/policy_server.py` | `deploy_octo` | `.venv` ✓ | **Deploy** (A100 verified) |
| **Diffusion Policy** | DDPM denoiser | PyTorch | `diffusion_policy/policy_server.py` | `deploy_diffusion_policy` | `.venv` ✓ | **Deploy** |
| **VQ-BeT** | Quantized transformer | PyTorch | `vq_bet/policy_server.py` | auto-discovered | `.venv` ✓ | **Import** ✓ |
| **BESO** | Score-based diffusion | PyTorch | `beso/policy_server.py` | `deploy_beso` | `.venv` ✓ | **Deploy** |
| **3D Diffusion Policy** | Point cloud diffusion | PyTorch | `3D-Diffusion-Policy/policy_server.py` | auto-discovered | `.venv` ✓ | **Syntax** ✓ |
| **ACT** | CVAE + Chunking Transformer | PyTorch | `act/policy_server.py` | — | — | **Smoke** ✓ (zero-touch onboard) |
| **OpenPI (pi0)** | Flow-based VLA | JAX | `openpi/scripts/policy_server.py` | auto-discovered | `.venv` ✓ | **Syntax** ✓ |

**Validation levels** (strongest → weakest):
- **Benchmark**: Full eval with real checkpoint, quantitative results
- **Deploy**: Server launched on GPU node, healthcheck passes, inference runs
- **Import**: Module loads under repo venv, BasePolicy subclass detected
- **Smoke**: Instantiate with `_validate_only=True`, fake `infer()` returns correct shape
- **Syntax**: `py_compile` passes, adapter is valid Python

### Supported Benchmarks

| Platform | Tasks | Eval Client | Status |
|----------|-------|-------------|--------|
| **LIBERO** | 5 suites: spatial, object, goal, 10, 90 (10-90 tasks each) | `LIBERO/scripts/run_eval.py` | **Benchmark results** |
| **ManiSkill** | PickCube, StackCube, PegInsertion, PickSingleYCB, PushCube + any env | `ManiSkill/scripts/run_eval.py` | **Running** |
| **RoboTwin** | 50 dual-arm tasks (stack, handover, place, pick, etc.) | `RoboTwin/script/run_eval_ws.py` | **Running** |
| **RoboCasa** | 20+ kitchen tasks (bake, boil, brew, chop, wash, etc.) | `robocasa/scripts/run_eval.py` | Eval client ready |

## Project Structure

```
agentic/
├── README.md
├── OVERVIEW.md              # Detailed technical documentation
├── robot_agent/             # Agent brain
│   ├── agent.py             # ReAct orchestrator (Claude/GPT/Qwen) + loop-breaker
│   ├── SOUL.md              # Agent identity and rules
│   └── skills/              # 36 SKILL.md-based skills
│       ├── analyze_repo/        # Clone + scan structure
│       ├── probe_run/           # Run demo, capture tensor shapes
│       ├── extract_io_spec/     # Parse probe output → IO spec
│       ├── infer_io_spec/       # 3-source merge (README/probe/user)
│       ├── onboard_benchmark/   # Simulator probes (libero/maniskill/...)
│       ├── check_finetune_capability/
│       ├── read_file/           # Inspect file with line numbers
│       ├── list_files/          # Glob-aware listing
│       ├── write_file/          # Whole-file write
│       ├── edit_file/           # Targeted base64 patch + auto-repair
│       ├── validate_policy_server/  # Sandboxed import/smoke check
│       ├── setup_env/           # uv venv + deps
│       ├── fix_deps/            # 15+ auto-fix patterns
│       ├── build_container/     # Apptainer builder
│       ├── download_model/      # Variant-aware HF download
│       ├── download_dataset/
│       ├── validate_dataset/    # lerobot/parquet/rlds/zarr detect
│       ├── generate_dataloader/
│       ├── validate_dataloader/ # Shape/finite gate before finetune
│       ├── run_benchmark/       # Multi-platform eval (SLURM/local)
│       ├── finetune/
│       ├── train_and_eval/      # SLURM: finetune → deploy → eval
│       ├── check_cluster_status/
│       ├── deploy_policy/       # Unified deploy (replaces 6 legacy)
│       ├── deploy_openvla/ ...  # Legacy per-model deploy skills
│       ├── test_policy_connection/
│       ├── stop_policy_server/
│       ├── generate_run_demo/       # Emit standalone run_demo.sh
│       ├── generate_run_evaluation/ # Emit standalone run_evaluation.sh
│       ├── create_deploy_skill/
│       ├── wrap_policy/         # Deprecated regex fallback
│       └── ...
├── policy_websocket/        # Universal WebSocket bridge
│   └── src/policy_websocket/
│       ├── base_policy.py       # 13-line ABC interface
│       ├── websocket_server.py  # Async server + msgpack
│       ├── websocket_client.py  # Client with auto-reconnect
│       ├── msgpack_numpy.py     # numpy binary serialization
│       ├── eval_preflight.py    # Registry gate (READY/NEEDS_FINETUNE/CROSS/UNSUP)
│       ├── eval_registry.py     # model × benchmark compatibility matrix
│       └── action_checker.py    # ActionSanityChecker (catches t=0 zero-init)
├── scripts/                 # Eval scripts, debug tools
└── containers/              # Apptainer .def files
```

## Why Skills Are Effective

Skills bridge the gap between LLM reasoning and system execution:

1. **LLM sees skills as tools**: `skill_to_claude_tool()` converts SKILL.md YAML into JSON Schema tool definitions. The LLM can "call" skills by providing parameter values.

2. **Skills are executable**: `command_template` is a real shell script with `{param}` placeholders. `execute_skill()` substitutes parameters and runs via `subprocess.run(["bash", "-c", cmd])`.

3. **Self-expanding**: `create_deploy_skill` generates new SKILL.md files. On next agent run, new skills are auto-loaded. The system grows its own capabilities.

4. **Auto-discovery**: Repos with `policy_server.yaml` are automatically converted to deploy skills at startup via `discover_policy_servers()`.

This means: **the LLM controls everything through a standard tool-calling API, while each skill is a self-contained, testable shell script**.

## License

MIT
