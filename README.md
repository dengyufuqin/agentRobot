# AgentRobot: One-Sentence Deployment for Robot Learning

An agentic orchestration system for robot learning on HPC clusters. Inspired by [OpenClaw](https://github.com/robot-claw/openclaw), AgentRobot enables **one-sentence deployment** — give a natural language command, and the system automatically handles everything: downloading models, setting up environments, deploying servers, running benchmarks, and reporting results.

## Core Capability

```
User: "用openvla跑LIBERO的benchmark"
Agent: check_cluster_status → run_benchmark(policy="openvla", benchmark="libero_spatial") → auto-resolves checkpoint
       ✅ 98% avg success rate (per-suite checkpoints auto-selected from eval_registry)

User: "集成 https://github.com/CleanDiffuserTeam/CleanDiffuser"
Agent: analyze_repo → setup_env → list_files → read_file → write_file (plaintext)
       → validate_policy_server → fix_deps → validate ✓
       ✅ 11步全自动，0人工干预，包含自愈
```

## End-to-End Auto-Deploy: Verified on 3 Fresh Repos

| Repo | Type | Skill calls | Outcome |
|---|---|:---:|---|
| **act** (existing server, no venv) | ACT (Action Chunking Transformer) | 3 | ✅ first-pass — setup_env → list_files → validate |
| **dobb-e** (fresh, no server, no venv) | Imitation-in-Homes BC | 7 | ✅ self-healed — write adapter → validate fail → fix_deps(msgpack) → validate ✓ |
| **cleandiffuser** (clean clone) | Diffusion-policy library | 11 | ✅ self-healed — recovered from wrong filename, then fix_deps + retry |

**Success rate: 3/3 = 100%** on previously-unseen repos with auto recovery from missing deps and minor LLM mistakes.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                        SOUL.md                               │
│             (Agent identity, rules, examples)                │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│                     agent.py                                  │
│      ReAct Loop (Qwen/Claude/GPT) + JSON容错 + 交互式断点     │
│      36 Skills + Auto-discovery from policy_server.yaml       │
│      Auto-encodes plaintext → b64 (LLM never touches base64)  │
│      Exit 10/3 → user picks variant or supplies fallback      │
└──────┬─────────┬──────────┬──────────┬───────────┬───────────┘
       │         │          │          │           │
   ┌───▼──┐  ┌──▼───┐  ┌──▼───┐  ┌──▼────┐  ┌──▼─────┐
   │ Meta │  │ Env  │  │ Eval │  │Deploy │  │ Util   │
   │Skills│  │Skills│  │Skills│  │Skills │  │Skills  │
   └──────┘  └──────┘  └──────┘  └───────┘  └────────┘
```

## 36 Skills

| Category | Skill | Description |
|----------|-------|-------------|
| **Meta / IO** | `analyze_repo` | GitHub URL **or** local path / `file://` — clone if needed, then deep-scan model class, deps, frameworks |
| **Meta / IO** | `list_files` | Glob a repo (`*.py`, etc.) — preferred entry point for unknown layouts |
| **Meta / IO** | `read_file` | Read range of lines, supports `start`/`end` |
| **Meta / IO** | `edit_file` | Atomic `old_string → new_string` (PREFERRED over write_file for small fixes) |
| **Meta / IO** | `write_file` | Whole-file write — pass `content` plaintext (agent auto-encodes to base64) |
| **Probe & Spec** | `probe_run` | Live model load + monkey-patched hook to capture obs/action shapes |
| **Probe & Spec** | `infer_io_spec` | 3-source merge: README regex + probe + user-fallback JSON |
| **Probe & Spec** | `extract_io_spec` | Parse `.probe_io_spec.json` → image_keys / state_keys / action_dim |
| **Probe & Spec** | `onboard_benchmark` | Symmetric to probe_run but for SIMULATOR repos (libero / maniskill / robocasa / calvin / simpler), supports `--lite` for no-GPU login nodes |
| **Env** | `setup_env` | uv venv + dep install, CUDA auto-detect, `--smoke` post-install validation |
| **Env** | `fix_deps` | Iterative diagnose-and-fix loop, 15+ error patterns, 50+ name mappings |
| **Env** | `build_container` | Apptainer `.def` + `.sif` (verified: `--fakeroot` works on login node) |
| **Adapter Generation** | `wrap_policy` | *(Deprecated; kept as regex fallback)* |
| **Adapter Generation** | `create_deploy_skill` | Auto-generate a new `deploy_*` SKILL.md |
| **Adapter Generation** | `generate_dataloader` | Emit a LeRobot-format dataloader stub (with v0.5/legacy import fallback) |
| **Adapter Generation** | `generate_run_demo` | Standalone bash runner that boots the policy server with venv |
| **Adapter Generation** | `generate_run_evaluation` | Orchestration script: server → port-wait → benchmark client → cleanup |
| **Validation** | `validate_policy_server` | Syntax + import + smoke check on an adapter (no model load) |
| **Validation** | `validate_dataset` | LeRobot codebase_version, meta files, parquet structure |
| **Validation** | `validate_dataloader` | Iterate the generated dataloader, capture batch shapes / structured errors |
| **Data / Model** | `download_model` | HF snapshot_download with **variant disambiguation** (exit 10 → interactive pick) |
| **Data / Model** | `download_dataset` | HF dataset pull, restrict via `--allow-patterns` |
| **Training** | `check_finetune_capability` | Scan repo for `train*.py`, README sections, console scripts before re-writing |
| **Training** | `finetune` | Wrapper for repo-native finetune entry point |
| **Training** | `train_and_eval` | Composite: train → eval-loop on a held-out split |
| **Cluster** | `check_cluster_status` | SLURM queue + GPU usage + free-node finder |
| **Eval** | `run_benchmark` | End-to-end: start server → run eval → report (3 modes: SLURM / existing / local) |
| **Deploy** | `deploy_policy` | Unified deploy entry point — replaces per-model deploy_* |
| **Deploy** | `deploy_{lerobot, openvla, octo, beso, diffusion_policy, robomimic, dp3, openpi, vq_bet}` | Per-repo deploy stubs (auto-discovered from `policy_server.yaml`) |
| **Test** | `test_policy_connection` | Verify a running server via WebSocket |
| **Test** | `stop_policy_server` | Stop a running policy server process |

## Standard Pipeline

```
        ┌─ existing repo + server ─┐         ┌─ fresh repo (no server) ─┐
        │                          │         │                          │
   [list_files]               [setup_env]   [analyze_repo / list_files]
        ↓                          ↓                     ↓
[validate_policy_server]      [validate_*]         [setup_env]
        ↓                                               ↓
       ✅                                       [read_file × N]  (find model + template)
                                                       ↓
                                                 [write_file]    (`content` plaintext)
                                                       ↓
                                              [validate_policy_server]
                                                       ↓
                                                  fail? → [fix_deps]
                                                       ↓
                                              [validate_policy_server]  retry
                                                       ↓
                                                       ✅
```

**Branches (on demand):**
- Need obs/action shapes → `probe_run` → `infer_io_spec` (3-source merge)
- Need benchmark wiring → `onboard_benchmark`
- Need training → `check_finetune_capability` → `finetune` or `train_and_eval`
- Need data → `download_dataset` → `validate_dataset` → `generate_dataloader` → `validate_dataloader`
- Need shippable scripts → `generate_run_demo` / `generate_run_evaluation`
- Need isolation → `build_container` (Apptainer)

**Interactive break-points:**
- Skill exits **10** → agent lists candidates, asks user to pick
- Skill exits **3** → agent prompts user to supply a fallback JSON path
- Non-TTY mode → skips prompt, falls back to LLM decision

## Verified End-to-End Flows

### Flow 1: One-Sentence Benchmark

**Input:** `"用openvla跑LIBERO-spatial的benchmark，提交SLURM任务到集群"`

**Agent execution (fully autonomous):**
1. `check_cluster_status()` → Found cn18 idle with 8x H100
2. `run_benchmark(policy="openvla", benchmark="libero_spatial", submit=true)` → Auto-resolved checkpoint from eval_registry, submitted SLURM job

**Result:** Job runs on H100, server + eval in same SLURM job. Verified: **98% success rate** (49/50 episodes). Checkpoint, unnorm_key, and per-suite config automatically selected — zero manual configuration.

### Flow 2: One-Sentence New Repo Integration (modern read→write→validate loop)

**Input:** `"集成 CleanDiffuser at /mnt/.../cleandiffuser"`

**Agent execution (fully autonomous, 11 steps, with self-recovery):**
1. `analyze_repo(repo_url="file:///mnt/.../cleandiffuser")` → file:// detected, skips clone, scans structure
2. `setup_env(repo_path=...)` → uv venv + installs torch + project deps
3. `read_file(...wrong_filename.py)` → returns `[ERROR] does not exist`
4. `list_files(pattern="*.py")` → agent recovers, finds correct snake_case file
5. `read_file(.../base_nn_diffusion.py)` → identifies `BaseNNDiffusion` class
6. `read_file(/mnt/.../diffusion_policy/policy_server.py)` → reference template
7. `write_file(content="<plaintext python code>")` → 55-line adapter (agent auto-encodes b64)
8. `validate_policy_server(mode=import)` → `OK: syntax`, `IMPORT_FAIL` (missing dep)
9. `fix_deps(repo_path=...)` → installs missing modules via `uv pip`
10. `validate_policy_server(mode=import)` → `OK: import` ✅
11. Agent reports completion

**Result:** Fresh GitHub clone → working policy_server.py with import-clean validation, **no human intervention**.

### Flow 3: Intelligent Error Recovery

**Input:** `"用openvla在LIBERO-spatial上跑benchmark评测，帮我找个可用的GPU节点"`

**Agent problem-solving (15-turn ReAct):**
1. Found cn30 idle → SSH denied (no SLURM job on node)
2. Tried cn19 → Same issue
3. Switched to cn06 (has active job, SSH allowed) → ✅ Deployed
4. Tested connection → Timeout (model loading)
5. Retried test → ✅ Connected, 747ms inference
6. Ran benchmark → Port conflict (server already running)
7. Stopped server → Retried → Login node has no GPU
8. Tried setup_env + fix_deps to debug
9. **Conclusion:** Agent correctly diagnosed each failure and adapted strategy

## Technical Components

### policy_websocket (Communication Layer)

The universal bridge between any model and any client:

```python
# Server side (13-line ABC)
class BasePolicy(ABC):
    def infer(self, obs: dict) -> dict: ...
    def reset(self): ...

# Auto-generated adapter wraps any model
class OpenVLAPolicy(BasePolicy):
    def infer(self, obs):
        action = self.model.predict_action(obs["image"])
        return {"actions": action}

# Client side
policy = WebsocketClientPolicy(host="cn19", port=18800)
result = policy.infer({"image": img, "task_description": "pick up the bowl"})
```

- Async WebSocket + msgpack binary serialization
- Healthcheck endpoint at `/healthz`
- Used by all 8+ integrated repos

### Smart Wrapper Generator (wrap_policy)

Analyzes repo source code and generates policy_server.py adapters:

| Pattern | Detection | Example Repo |
|---------|-----------|-------------|
| Hydra/OmegaConf | `@hydra.main`, `OmegaConf.load` | diffusion_policy, BESO |
| from_pretrained | `.from_pretrained()` | OpenVLA, HuggingFace models |
| algo_factory | `algo_factory()`, `.deserialize()` | robomimic/DROID |

### fix_deps (Dependency Auto-Repair)

Iterative diagnosis loop: test imports → match error → apply fix → retry.

**15+ error patterns:**
| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: No module named 'X'` | `uv pip install X` (with 50+ name mappings) |
| `ImportError: libGL.so.1` | Switch to `opencv-python-headless` |
| numpy 2.x incompatibility | Pin `numpy<2` (preemptive detection) |
| mujoco/robosuite crash | Pin `mujoco==2.3.7` |
| `torch` has no attribute `xpu` | Pin `diffusers<0.27` |
| cmake required | Install `cmake` |
| Repeated failure (e.g., pytorch3d) | Stop retrying, report as manual fix needed |

**Features:**
- Preemptive numpy 2.x detection before any import fails
- Auto-detects nested repo structures (PYTHONPATH)
- Duplicate failure detection (avoids infinite retry loops)
- Integrated into `setup_env` as post-install step

### run_benchmark (End-to-End Evaluation)

**One-sentence deployment** — checkpoint, unnorm_key, and server_args auto-resolved from `eval_registry`:
```bash
python run_benchmark.py --policy openvla --benchmark libero_spatial  # that's it
python run_benchmark.py --policy pi0.5 --benchmark libero_10
```

**Preflight safety:** Blocks known-bad combos (NEEDS_FINETUNE, CROSS_DOMAIN) before wasting GPU hours.

Three execution modes:

| Mode | Use Case | How It Works |
|------|----------|-------------|
| **SLURM Submit** (default) | HPC cluster | Generates sbatch script, submits job. Server + eval in one GPU job |
| **Existing Server** | Server already deployed | Connects to `host:port`, runs eval only |
| **Local** | Dev/debug on GPU node | Starts server + eval in same process |

### agent.py (ReAct Orchestrator)

- Supports Anthropic (Claude), OpenAI (GPT), Dashscope (Qwen) APIs
- **JSON error tolerance:** Auto-repair truncated JSON from LLM + retry with error feedback
- **15-turn max** ReAct loop with tool call chaining
- Auto-discovers repos with `policy_server.yaml`
- Converts SKILL.md YAML frontmatter to LLM tool definitions

## Integrated Algorithms (12)

| Model | Type | Framework | Benchmarks Tested | Status |
|-------|------|-----------|-------------------|--------|
| **OpenVLA-OFT 7B** | Vision-Language-Action | PyTorch | LIBERO (4), ManiSkill (7), RoboCasa (6) | ✅ **98%** LIBERO avg |
| **pi0.5** (LeRobot) | Flow-matching VLA | PyTorch | LIBERO (4), ManiSkill (7), RoboCasa (6) | ✅ **92%** LIBERO avg |
| **pi0** (LeRobot) | Flow-matching VLA | PyTorch | LIBERO (4), ManiSkill (1), RoboCasa (6) | ✅ **65%** LIBERO avg |
| **SmolVLA** (LeRobot) | Small VLA | PyTorch | LIBERO (4), ManiSkill (7), RoboCasa (6) | ✅ Benchmarked |
| **Octo-small** | Transformer policy | JAX | LIBERO (4), ManiSkill (7), RoboCasa (6) | ✅ Benchmarked |
| **SpatialVLA-4B** | Spatial VLA | PyTorch | LIBERO (4), ManiSkill (7), RoboCasa (6) | ✅ Benchmarked |
| **RDT-1B** | Diffusion Transformer | PyTorch | ManiSkill (5) | ✅ Integrated |
| **Diffusion Policy** | DDPM denoiser | PyTorch | — | ✅ Integrated |
| **VQ-BeT** | Quantized transformer | PyTorch | — | ✅ Integrated |
| **BESO** | Score-based diffusion | PyTorch | — | ✅ Auto-integrated by Agent |
| **ACT** | Action Chunking Transformer | PyTorch | — | ✅ Integrated |
| **3D Diffusion Policy** | Point cloud diffusion | PyTorch | — | ⚠️ Partial (pytorch3d issue) |

## Supported Benchmarks (4)

| Benchmark | Type | Tasks | Robot | Status |
|-----------|------|-------|-------|--------|
| **LIBERO** | Tabletop manipulation | 4 suites (spatial, object, goal, long-horizon) × 10 tasks | Franka Panda | ✅ Fully tested |
| **ManiSkill** | Diverse manipulation | 7 tasks (Pick, Stack, Push, Peg, Faucet, Lift, Plug) | Franka Panda | ✅ Fully tested |
| **RoboCasa** | Kitchen manipulation | 6 tasks (PnP, Door, Faucet) | PandaMobile | ✅ Setup complete |
| **RoboTwin** | Dual-arm manipulation | 5+ tasks | Dual-arm | ⚠️ Requires Vulkan |

## Benchmark Results

All evaluations run on NVIDIA H100 80GB HBM3 via SLURM. The **same `policy_websocket` protocol** connects every algorithm to every benchmark — no per-combination glue code.

### LIBERO Benchmark (Success Rate %, 10 tasks × 5 trials per suite)

| Algorithm | Checkpoint | Spatial | Object | Goal | LIBERO-10 | Avg |
|-----------|-----------|:------:|:------:|:----:|:---------:|:---:|
| **OpenVLA-OFT** (per-suite ckpt) | openvla-7b-oft-finetuned-libero-{suite} | **98** | **100** | **94** | **100** | **98.0** |
| **pi0.5** | lerobot/pi05_libero_finetuned | 88 | 88 | **96** | 96 | **92.0** |
| **pi0** | lerobot/pi0_libero_finetuned | 60 | 80 | 78 | 42 | **65.0** |

**Key findings (v11 — correct checkpoints + image flip + per-suite resolution):**
- **OpenVLA-OFT achieves 98% average** across all 4 suites — 100% on object and LIBERO-10 (50/50 episodes each)
- **pi0.5 is the best single-checkpoint model** at 92% average — one checkpoint handles all 4 suites
- **pi0 achieves 65% average** — solid generalization from a single checkpoint, but below pi0.5
- Earlier versions (v9/v10) showed 0% for pi0.5 and pi0 due to wrong checkpoint names (`lerobot/pi0_libero` → base model, not finetuned) and missing image flip (LeRobot applies 180° rotation during data collection)
- Per-suite checkpoints for OpenVLA are essential: a single spatial checkpoint gets 100%/0%/6%/8% across suites

### ManiSkill Benchmark (Success Rate %, 5 trials per task)

| Algorithm | PickCube | StackCube | PushCube | PegInsertion | TurnFaucet | LiftPeg | PlugCharger |
|-----------|:--------:|:---------:|:--------:|:------------:|:----------:|:-------:|:-----------:|
| **OpenVLA-OFT** (unified) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **OpenVLA-OFT** (spatial) | — | — | — | 0 | 0 | 0 | 0 |
| **pi0** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **pi0.5** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **Octo** | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **SpatialVLA** | 0 | 0 | 0 | 0 | *...* | *...* | *...* |
| **SmolVLA** | *...* | *...* | *...* | *...* | *...* | *...* | *...* |

- All LIBERO-finetuned models get 0% on ManiSkill — **expected** (different embodiment, obs format, action space)
- SpatialVLA (pre-trained, not ManiSkill-finetuned) also 0% — confirms cross-domain gap
- This validates our cross-benchmark infrastructure works correctly (pipeline runs, just no transfer)
- SmolVLA ManiSkill evaluation in progress

### RoboCasa Kitchen Benchmark (6 tasks × 5 trials)

| Algorithm | PnPCounterToCab | PnPCabToCounter | PnPCounterToSink | OpenDoor | CloseDoor | TurnFaucet |
|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **pi0** | 0 | 0 | *...* | *...* | *...* | *...* |
| **pi0.5** | 0 | *...* | *...* | *...* | *...* | *...* |
| **OpenVLA-OFT** | 0 | 0 | 0 | *...* | *...* | *...* |
| **Octo** | 0 | 0 | 0 | *...* | *...* | *...* |
| **SmolVLA** | *...* | *...* | *...* | *...* | *...* | *...* |
| **SpatialVLA** | *...* | *...* | *...* | *...* | *...* | *...* |

*RoboCasa evaluation in progress — all 6 algorithm jobs running on cluster, partial results shown above (all 0% so far). These models are LIBERO-finetuned, not RoboCasa-finetuned.*

**Key insight:** The evaluation infrastructure is **model-agnostic and benchmark-agnostic**. Adding a new algorithm or benchmark requires only a thin `BasePolicy` adapter (~50 lines) — the WebSocket bridge, SLURM orchestration, and eval harness are fully reusable. We are testing **7 algorithms × 3 benchmarks = 21 combinations** through the same pipeline, with 10 jobs running concurrently on the cluster.

## Apptainer Containers

| Container | Base | Size | Status |
|-----------|------|------|--------|
| `policy_base.sif` | python:3.11-slim | 78 MB | ✅ Verified |
| `diffusion_policy.sif` | NGC PyTorch 24.01 | 13 GB | ✅ GPU verified on H100 |

## Cluster Environment

- **Cluster:** hessian.AI 43, TU Darmstadt
- **Nodes:** 35 compute nodes (cn01–cn35), each with 8x NVIDIA H100 80GB HBM3
- **Scheduler:** SLURM
- **Storage:** VAST filesystem at `/mnt/vast/`
- **Tools:** `uv` (fast Python package manager), Apptainer (container runtime)

## Project Structure

```
agentRobot/
├── agentic/                  # Our work — agent + communication layer
│   ├── OVERVIEW.md           # Detailed technical documentation
│   ├── robot_agent/          # Agent brain
│   │   ├── agent.py          # ReAct orchestrator (Qwen/Claude/GPT)
│   │   ├── SOUL.md           # Agent identity and rules
│   │   └── skills/           # 19 SKILL.md-based skills
│   │       ├── run_benchmark/    # End-to-end eval (SLURM/local)
│   │       ├── fix_deps/         # Auto-diagnose and fix deps
│   │       ├── wrap_policy/      # Smart policy_server.py generator
│   │       ├── analyze_repo/     # Clone and analyze GitHub repos
│   │       ├── setup_env/        # venv + dependency installation
│   │       ├── deploy_openvla/   # OpenVLA deployment
│   │       └── ...               # 12 more skills
│   ├── policy_websocket/     # Universal WebSocket bridge
│   │   └── src/policy_websocket/
│   │       ├── base_policy.py        # 13-line ABC
│   │       ├── websocket_server.py   # Async server + msgpack
│   │       ├── websocket_client.py   # Client with auto-reconnect
│   │       └── msgpack_numpy.py      # numpy binary serialization
│   ├── scripts/              # Eval scripts, debug tools
│   └── containers/           # Apptainer .def files
│
├── openvla/                  # OpenVLA-OFT 7B (PyTorch)
├── lerobot/                  # pi0 / pi0.5 / SmolVLA (LeRobot framework)
├── octo/                     # Octo model (JAX)
├── SpatialVLA/               # SpatialVLA-4B (HuggingFace transformers)
├── RDT/                      # RDT-1B Diffusion Transformer
├── openpi/                   # pi0 VLA model (JAX)
├── diffusion_policy/         # DDPM policy
├── vq_bet/                   # VQ-BeT (auto-integrated)
├── beso/                     # BESO (auto-integrated by Agent)
├── act/                      # ACT (Action Chunking Transformer)
├── 3D-Diffusion-Policy/      # DP3 (partial)
├── droid_policy_learning/    # DROID dataset training
│
├── LIBERO/                   # 130+ manipulation benchmark
├── ManiSkill/                # ManiSkill sim benchmark (SAPIEN)
├── robocasa/                 # RoboCasa kitchen benchmark
├── RoboTwin/                 # RoboTwin dual-arm benchmark
├── SimplerEnv/               # SimplerEnv real-to-sim eval
├── CALVIN_bench/             # CALVIN language-conditioned benchmark
└── logs/                     # SLURM job logs, eval results
```

## Usage

### Interactive Mode
```bash
export DASHSCOPE_API_KEY=sk-xxx  # or ANTHROPIC_API_KEY, OPENAI_API_KEY
python agentic/robot_agent/agent.py
# You> 用openvla跑LIBERO-spatial
# You> 集成 https://github.com/xxx/new-model
# You> 检查集群哪些节点有空闲GPU
```

### Single Command Mode
```bash
python agentic/robot_agent/agent.py "在cn19上部署openvla并测试连接"
```

### Direct Skill Execution
```bash
# One-sentence benchmark (checkpoint auto-resolved from eval_registry)
python agentic/robot_agent/skills/run_benchmark/run_benchmark.py \
  --policy openvla --benchmark libero_spatial --submit

# With explicit checkpoint
python agentic/robot_agent/skills/run_benchmark/run_benchmark.py \
  --policy pi0.5 --checkpoint lerobot/pi05_libero_finetuned \
  --benchmark libero_10 --num_trials 5 --node cn19 --submit

# Fix dependencies
python agentic/robot_agent/skills/fix_deps/fix_deps.py /path/to/repo --max-retries 5

# Generate policy adapter
python agentic/robot_agent/skills/wrap_policy/generate_smart.py /path/to/repo ModelClass module.path
```

## Comparison with OpenClaw

| Feature | OpenClaw | AgentRobot |
|---------|----------|------------|
| Skill system | SKILL.md + YAML | ✅ Same pattern |
| Self-expanding skills | ✅ | ✅ create_deploy_skill |
| Policy abstraction | BasePolicy ABC | ✅ policy_websocket |
| Smart code generation | — | ✅ 3-pattern model detection |
| Dependency auto-repair | — | ✅ fix_deps (15+ patterns) |
| One-sentence benchmark | — | ✅ run_benchmark (SLURM) |
| Container support | Docker | ✅ Apptainer (.sif for HPC) |
| Multi-model LLM backend | Claude only | ✅ Claude/GPT/Qwen |
| JSON error tolerance | — | ✅ Auto-repair + retry |
| HPC cluster native | — | ✅ SLURM, multi-node, H100 |
