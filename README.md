# AgentRobot: One-Sentence Deployment for Robot Learning

An agentic orchestration system for robot learning on HPC clusters. Inspired by [OpenClaw](https://github.com/robot-claw/openclaw), AgentRobot enables **one-sentence deployment** — give a natural language command, and the system automatically handles everything: downloading models, setting up environments, deploying servers, running benchmarks, and reporting results.

## Core Capability

```
User: "用openvla跑LIBERO-spatial的benchmark"
Agent: check_cluster_status → run_benchmark(submit=SLURM) → Job running on H100
       ✅ 94.0% success rate (47/50 episodes)

User: "集成 https://github.com/intuitive-robots/beso"
Agent: analyze_repo → setup_env → fix_deps → wrap_policy → create_deploy_skill
       ✅ 5步全自动，0人工干预
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
│         ReAct Loop (Qwen/Claude/GPT) + JSON容错               │
│         19 Skills + Auto-discovery from policy_server.yaml    │
└──────┬─────────┬──────────┬──────────┬───────────┬───────────┘
       │         │          │          │           │
   ┌───▼──┐  ┌──▼───┐  ┌──▼───┐  ┌──▼────┐  ┌──▼─────┐
   │ Meta │  │ Env  │  │ Eval │  │Deploy │  │ Util   │
   │Skills│  │Skills│  │Skills│  │Skills │  │Skills  │
   └──────┘  └──────┘  └──────┘  └───────┘  └────────┘
```

## 19 Skills

| Category | Skill | Description |
|----------|-------|-------------|
| **Meta** | `analyze_repo` | Clone GitHub URL, analyze model structure and inference patterns |
| **Meta** | `wrap_policy` | Smart generator: 3 patterns (Hydra, from_pretrained, algo_factory) |
| **Meta** | `create_deploy_skill` | Auto-generate new deploy SKILL.md (self-expanding) |
| **Meta** | `write_file` | Write arbitrary file content |
| **Env** | `setup_env` | Create venv, install deps, CUDA auto-detect, pin torch cu121 |
| **Env** | `fix_deps` | Auto-diagnose and fix dependency issues (15+ patterns, 50+ mappings) |
| **Env** | `build_container` | Generate Apptainer .def files, build .sif containers |
| **Eval** | `run_benchmark` | End-to-end: start server → run eval → report results (3 modes) |
| **Cluster** | `check_cluster_status` | Query SLURM jobs, GPU usage, find available nodes |
| **Deploy** | `deploy_openvla` | Deploy OpenVLA-OFT on GPU node |
| **Deploy** | `deploy_octo` | Deploy Octo model |
| **Deploy** | `deploy_diffusion_policy` | Deploy Diffusion Policy |
| **Deploy** | `deploy_robomimic` | Deploy Robomimic models |
| **Deploy** | `deploy_beso` | Deploy BESO score-based diffusion |
| **Deploy** | `deploy_dp3` | Deploy 3D Diffusion Policy (auto-discovered) |
| **Deploy** | `deploy_openpi` | Deploy OpenPI/pi0 (auto-discovered) |
| **Deploy** | `deploy_vq_bet` | Deploy VQ-BeT (auto-discovered) |
| **Test** | `test_policy_connection` | Verify running server via WebSocket |
| **Deploy** | `stop_policy_server` | Stop running policy server process |

## Verified End-to-End Flows

### Flow 1: One-Sentence Benchmark

**Input:** `"用openvla跑LIBERO-spatial的benchmark，提交SLURM任务到集群"`

**Agent execution (fully autonomous):**
1. `check_cluster_status()` → Found cn19 idle with 8x H100
2. `run_benchmark(policy="openvla", benchmark="libero_spatial", node="cn19", submit=true)` → Submitted SLURM job

**Result:** Job runs on H100, server + eval in same SLURM job. Verified: **94.0% success rate** (47/50 episodes, ~59ms/step inference, ~11min total on H100).

### Flow 2: One-Sentence New Repo Integration

**Input:** `"帮我集成 https://github.com/intuitive-robots/beso"`

**Agent execution (fully autonomous, 5 steps):**
1. `analyze_repo(repo_url="https://github.com/intuitive-robots/beso")` → Cloned, identified BesoAgent model class
2. `setup_env(repo_path=".../beso")` → Created venv, installed torch cu121 + beso package
3. `fix_deps(repo_path=".../beso")` → Auto-fixed msgpack, websockets, pinned numpy<2
4. `wrap_policy(model_class="BesoAgent", framework="torch")` → Generated 115-line policy_server.py
5. `create_deploy_skill(skill_name="deploy_beso")` → Created deploy skill with GPU/port management

**Result:** New repo fully integrated, `deploy_beso` skill available immediately.

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

## Integrated Models

| Model | Type | Framework | Benchmarks Tested | Status |
|-------|------|-----------|-------------------|--------|
| OpenVLA-OFT 7B | Vision-Language-Action | PyTorch | LIBERO (4 suites), ManiSkill (3 tasks) | ✅ 94% LIBERO-spatial |
| pi0.5 (LeRobot) | Flow-matching VLA | PyTorch | LIBERO (4 suites), ManiSkill (3 tasks) | ✅ Benchmarked |
| Octo-small | Transformer policy | JAX | LIBERO (4 suites), ManiSkill (3 tasks) | ✅ Benchmarked |
| OpenPI (pi0) | Flow-based VLA | JAX | — | ✅ Server verified |
| Diffusion Policy | DDPM denoiser | PyTorch | — | ✅ Integrated |
| Robomimic | Behavior cloning | PyTorch | — | ✅ Integrated (factory pattern) |
| VQ-BeT | Quantized transformer | PyTorch | — | ✅ All imports pass |
| BESO | Score-based diffusion | PyTorch | — | ✅ Auto-integrated by Agent |
| 3D Diffusion Policy | Point cloud diffusion | PyTorch | — | ⚠️ Partial (pytorch3d issue) |

## Benchmark Results

All evaluations run on NVIDIA H100 80GB HBM3 via SLURM. The **same `policy_websocket` protocol** connects every algorithm to every benchmark — no per-combination glue code.

### Cross-Benchmark Matrix (Success Rate %)

| Algorithm | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal | LIBERO-10 | ManiSkill PickCube | ManiSkill StackCube | ManiSkill PushCube |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **OpenVLA-OFT 7B** | **94.0** | **82.0** | **86.0** | **58.0** | 0.0 | 0.0 | 0.0 |
| **pi0.5 (v0.44)** | 2.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| **Octo-small** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

- LIBERO: 50 episodes per suite (10 tasks x 5 trials). ManiSkill: 5 episodes per task.
- OpenVLA-OFT is fine-tuned on LIBERO; pi0.5 and Octo use pretrained/community checkpoints.
- ManiSkill 0% is expected for LIBERO-finetuned models (different embodiment/domain).
- RoboTwin evaluation requires Vulkan (SAPIEN renderer) — not available on headless HPC nodes.

**Key insight:** The evaluation infrastructure is **model-agnostic and benchmark-agnostic**. Adding a new algorithm or benchmark requires only a thin `BasePolicy` adapter (~50 lines) — the WebSocket bridge, SLURM orchestration, and eval harness are fully reusable.

### OpenVLA-OFT per-task breakdown (LIBERO-Spatial, 5 trials/task)

| Task | Description | Success Rate |
|------|-------------|:---:|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate | 100% |
| 1 | pick up the black bowl next to the cookie box and place it on the plate | 100% |
| 2 | pick up the black bowl on the cookie box and place it on the plate | 80% |
| 3 | pick up the black bowl next to the ramekin and place it on the stove | 80% |
| 4 | pick up the black bowl from the top of the cabinet and place it on the plate | 80% |
| 5 | pick up the black bowl next to the plate and place it on the plate | 100% |
| 6 | pick up the ketchup from the left of the stove and place it on the counter | 100% |
| 7 | pick up the black bowl on the stove and place it on the counter | 100% |
| 8 | pick up the black bowl in the top drawer of the cabinet and place it on the counter | 100% |
| 9 | pick up the black bowl on the wooden tray and place it on the plate | 100% |
| **Overall** | | **94.0%** |

- Paper reports 96.7% (50 trials) — our 94.0% (5 trials) is consistent
- Inference: ~59ms/step on H100

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
├── openvla/                  # OpenVLA-OFT 7B
├── lerobot/                  # pi0.5 / SmolVLA (LeRobot framework)
├── octo/                     # Octo model (JAX)
├── openpi/                   # pi0 VLA model (JAX)
├── LIBERO/                   # 130+ manipulation benchmark
├── ManiSkill/                # ManiSkill2 sim benchmark (SAPIEN)
├── RoboTwin/                 # RoboTwin dual-arm benchmark
├── diffusion_policy/         # DDPM policy
├── droid_policy_learning/    # DROID dataset training
├── vq_bet/                   # VQ-BeT (auto-integrated)
├── beso/                     # BESO (auto-integrated by Agent)
├── 3D-Diffusion-Policy/      # DP3 (partial)
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
# Run benchmark via SLURM
python agentic/robot_agent/skills/run_benchmark/run_benchmark.py \
  --policy openvla --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --benchmark libero_spatial --num_trials 5 --node cn19 --submit

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
