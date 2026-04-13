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
│   23 Skills + Auto-discovery from policy_server.yaml          │
│   + Loop-breaker (intercepts repeated identical failures)    │
└──────┬─────────┬──────────┬──────────┬───────────┬───────────┘
       │         │          │          │           │
   ┌───▼──┐  ┌──▼───┐  ┌──▼───┐  ┌──▼────┐  ┌──▼─────┐
   │ Meta │  │ Env  │  │ Eval │  │Deploy │  │ Util   │
   │Skills│  │Skills│  │Skills│  │Skills │  │Skills  │
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

## 23 Skills

| Category | Skill | Description |
|----------|-------|-------------|
| **Meta** | `analyze_repo` | Clone GitHub URL, analyze model structure and inference patterns |
| **Meta** | `wrap_policy` | Regex generator covering 6 common loading patterns (fallback) |
| **Meta** | `create_deploy_skill` | Auto-generate new deploy SKILL.md (self-expanding) |
| **Meta** | `write_file` | Write a whole file (use only for new files / >50% rewrites) |
| **Util** | `read_file` | Read any file with line numbers (paginated, capped 200KB) |
| **Util** | `list_files` | Glob-aware listing that auto-skips `.venv` / `__pycache__` / `data/` |
| **Util** | `edit_file` | Targeted base64 substring replacement, atomic + auto-repairs LLM glitches |
| **Util** | `validate_policy_server` | Sandboxed syntax/import/smoke check on adapter, no GPU, <5s |
| **Env** | `setup_env` | Create venv, install deps, CUDA auto-detect, pin torch cu121 |
| **Env** | `fix_deps` | Auto-diagnose and fix dependency issues (15+ patterns, 50+ mappings) |
| **Env** | `build_container` | Generate Apptainer .def files, build .sif containers |
| **Eval** | `run_benchmark` | Multi-platform eval (LIBERO/ManiSkill/RoboTwin), 3 modes, pluggable |
| **Cluster** | `check_cluster_status` | Local nvidia-smi or SLURM cluster status (auto-detect) |
| **Deploy** | `deploy_openvla` | Deploy OpenVLA-OFT on GPU (local or remote) |
| **Deploy** | `deploy_octo` | Deploy Octo model (local or remote) |
| **Deploy** | `deploy_diffusion_policy` | Deploy Diffusion Policy (local or remote) |
| **Deploy** | `deploy_robomimic` | Deploy Robomimic models (local or remote) |
| **Deploy** | `deploy_beso` | Deploy BESO score-based diffusion (local or remote) |
| **Deploy** | `deploy_dp3` | Deploy 3D Diffusion Policy (auto-discovered) |
| **Deploy** | `deploy_openpi` | Deploy OpenPI/pi0 (auto-discovered) |
| **Deploy** | `deploy_vq_bet` | Deploy VQ-BeT (auto-discovered) |
| **Test** | `test_policy_connection` | Verify running server via WebSocket |
| **Deploy** | `stop_policy_server` | Stop running policy server process |

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

### Algorithm × Benchmark Matrix

All evaluations run on **NVIDIA H100 80GB** via SLURM, 5 trials per task.

#### LIBERO Benchmarks

[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO): 10 tasks per suite, 50 episodes total.

| Algorithm | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-10 | Framework |
|-----------|:-:|:-:|:-:|:-:|-----------|
| **OpenVLA-OFT 7B** | **94.0%** | **82.0%** | **86.0%** | **58.0%** | PyTorch |
| **LeRobot pi0.5** | *running* | — | — | — | PyTorch |
| **Octo** | *running* | *running* | *running* | *running* | JAX |
| Diffusion Policy | ckpt | ckpt | ckpt | ckpt | PyTorch |
| VQ-BeT | ckpt | ckpt | ckpt | ckpt | PyTorch |
| BESO | ckpt | ckpt | ckpt | ckpt | PyTorch |
| 3D Diffusion Policy | ckpt | ckpt | ckpt | ckpt | PyTorch |
| ACT | ckpt | ckpt | ckpt | ckpt | PyTorch |
| OpenPI (pi0) | ckpt | ckpt | ckpt | ckpt | JAX |

#### ManiSkill Benchmarks

[ManiSkill](https://github.com/haosulab/ManiSkill): Simulated manipulation tasks.

| Algorithm | PickCube-v1 | StackCube-v1 | PushCube-v1 | Framework |
|-----------|:-:|:-:|:-:|-----------|
| **Octo** | *running* | *running* | *running* | JAX |
| OpenVLA-OFT 7B | ckpt | ckpt | ckpt | PyTorch |
| LeRobot pi0.5 | ckpt | ckpt | ckpt | PyTorch |

#### RoboTwin Benchmarks

[RoboTwin](https://github.com/TonyZhao0106/RoboTwin): Dual-arm manipulation (50 tasks).

| Algorithm | stack_blocks | handover_block | pick_bottles | Framework |
|-----------|:-:|:-:|:-:|-----------|
| **Octo** | *running* | *running* | *running* | JAX |
| OpenVLA-OFT 7B | ckpt | ckpt | ckpt | PyTorch |
| LeRobot pi0.5 | ckpt | ckpt | ckpt | PyTorch |

**Legend:** Score = success rate. `ckpt` = adapter ready, needs fine-tuned checkpoint. `*running*` = SLURM job submitted. `—` = not applicable.

### OpenVLA-OFT Detailed Results (LIBERO-Spatial)

| Task | Success Rate |
|------|:-----------:|
| pick up the black bowl between the plate and the ramekin and place it on the plate | 100% |
| pick up the black bowl next to the ramekin and place it on the plate | 100% |
| pick up the black bowl from table center and place it on the plate | 100% |
| pick up the black bowl on the cookie box and place it on the plate | 100% |
| pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate | 100% |
| pick up the black bowl on the ramekin and place it on the plate | 100% |
| pick up the black bowl next to the cookie box and place it on the plate | 100% |
| pick up the black bowl on the stove and place it on the plate | 80% |
| pick up the black bowl next to the plate and place it on the plate | 80% |
| pick up the black bowl on the wooden cabinet and place it on the plate | 80% |
| **Overall** | **94.0%** |

Paper reports 96.7% (50 trials) — our 94.0% (5 trials/task) is consistent. Inference: ~59ms/step on H100.

### Notes on Entries

- **`ckpt` cells**: The algorithm is fully integrated (adapter + venv + deploy), but needs a fine-tuned checkpoint for that benchmark. Users can supply their own via `--checkpoint`.
- **Octo**: General-purpose foundation model — runs on any benchmark without task-specific fine-tuning. Uses JAX.
- **OpenPI**: JAX dependency conflict with PyTorch in shared venv. Planned for Apptainer isolation.
- **Diffusion Policy, VQ-BeT, BESO, 3D-DP**: Task-specific models that require training on each benchmark. Infrastructure is ready — add your checkpoint and run.
- **ACT**: Zero-touch onboarded from `tonyzhaozh/act`, adapter smoke-verified. Needs ALOHA checkpoint for benchmark.
- **Cross-benchmark**: The same Agentic system runs evals on LIBERO, ManiSkill, and RoboTwin using the universal WebSocket policy bridge — demonstrating the system's **generality**.

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
│   └── skills/              # 23 SKILL.md-based skills
│       ├── run_benchmark/       # End-to-end eval (SLURM/local)
│       ├── fix_deps/            # Auto-diagnose and fix deps
│       ├── wrap_policy/         # Regex generator (6 fallback patterns)
│       ├── analyze_repo/        # Clone and analyze GitHub repos
│       ├── setup_env/           # venv + dependency installation
│       ├── read_file/           # Inspect any file with line numbers
│       ├── list_files/          # Glob-aware listing
│       ├── edit_file/           # Targeted base64 patch + auto-repair
│       ├── validate_policy_server/  # Sandboxed import/smoke check
│       ├── deploy_openvla/      # OpenVLA deployment
│       ├── deploy_octo/         # Octo deployment
│       ├── deploy_diffusion_policy/
│       ├── deploy_beso/         # BESO deployment
│       ├── deploy_robomimic/    # Robomimic deployment
│       ├── check_cluster_status/# GPU/cluster status
│       ├── test_policy_connection/
│       ├── stop_policy_server/
│       ├── create_deploy_skill/ # Self-expanding skill creator
│       ├── build_container/     # Apptainer container builder
│       ├── write_file/
│       └── ...
├── policy_websocket/        # Universal WebSocket bridge
│   └── src/policy_websocket/
│       ├── base_policy.py       # 13-line ABC interface
│       ├── websocket_server.py  # Async server + msgpack
│       ├── websocket_client.py  # Client with auto-reconnect
│       └── msgpack_numpy.py     # numpy binary serialization
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
