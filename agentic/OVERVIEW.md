# Agentic: One-Sentence Deployment for Robot Learning

一个基于 LLM 的自主编排系统，用于 HPC 集群上的机器人学习。用户说一句话，系统自动完成全部流程：下载模型、搭建环境、修复依赖、部署推理服务、提交评测任务、汇报结果。

**验证结果**: OpenVLA-OFT 在 LIBERO-Spatial 上 **94.0% 成功率** (47/50 episodes)，全流程由 Agent 自主完成。

---

## 目录

1. [系统总览](#1-系统总览)
2. [我们做了什么](#2-我们做了什么)
3. [核心代码清单](#3-核心代码清单)
4. [Skill 为什么有效 — LLM API 如何操控一切](#4-skill-为什么有效--llm-api-如何操控一切)
5. [完整 Pipeline：从用户打字到 94% 成功率](#5-完整-pipeline从用户打字到-94-成功率)
6. [通信协议：WebSocket + msgpack](#6-通信协议websocket--msgpack)
7. [模型推理：单步数据流](#7-模型推理单步数据流)
8. [验证过的完整流程](#8-验证过的完整流程)
9. [已集成模型](#9-已集成模型)
10. [目录结构](#10-目录结构)

---

## 1. 系统总览

### 三层架构

```
┌──────────────────────────────────────────────────────────────────┐
│  Layer 1: 大脑层 — agent.py + SOUL.md                            │
│  LLM (Claude/Qwen/GPT) ReAct 循环                               │
│  理解自然语言意图 → 选择 skill → 读取结果 → 决定下一步             │
├──────────────────────────────────────────────────────────────────┤
│  Layer 2: 技能层 — 19 个 SKILL.md                                │
│  每个 skill = YAML 参数声明 + bash/python 执行模板                │
│  LLM 看到的是"函数签名"，执行时展开为 shell 命令                   │
├──────────────────────────────────────────────────────────────────┤
│  Layer 3: 通信层 — policy_websocket (4 个文件, ~280 行)           │
│  BasePolicy ABC → WebSocket Server (async) → msgpack 二进制      │
│  所有模型统一接口: infer(obs) → actions                           │
└──────────────────────────────────────────────────────────────────┘
```

### 一句话概括

**agent.py 是大脑，skill 是手，policy_websocket 是神经（连接模型和评测），fix_deps/run_benchmark/wrap_policy 是最复杂的三只手。**

---

## 2. 我们做了什么

### 2.1 自己从零写的核心代码

| 组件 | 文件 | 行数 | 作用 |
|------|------|------|------|
| **Agent 大脑** | `robot_agent/agent.py` | 578 | ReAct 循环 + Skill 加载 + LLM 调用 + JSON 容错 |
| **Agent 身份** | `robot_agent/SOUL.md` | 93 | System prompt, 告诉 LLM 它是谁、有什么工具 |
| **WebSocket 服务端** | `policy_websocket/src/.../websocket_server.py` | 128 | async 接收 obs, 调 infer, 返 action |
| **WebSocket 客户端** | `policy_websocket/src/.../websocket_client.py` | 85 | 同步客户端, 给评测脚本用 |
| **策略基类** | `policy_websocket/src/.../base_policy.py` | 13 | 统一接口: `infer(obs) → action` |
| **序列化层** | `policy_websocket/src/.../msgpack_numpy.py` | 54 | numpy array 二进制传输 |
| **评测引擎** | `robot_agent/skills/run_benchmark/run_benchmark.py` | 428 | SLURM 提交 + 3 种评测模式 |
| **依赖自修复** | `robot_agent/skills/fix_deps/fix_deps.py` | ~500 | 15+ 错误模式, 50+ 包名映射, 迭代修复 |
| **适配器生成** | `robot_agent/skills/wrap_policy/generate_smart.py` | 200+ | 3 种模式检测, 自动生成 policy_server.py |
| **19 个 Skill** | `robot_agent/skills/*/SKILL.md` | 500+ | 从集群查询到模型部署的全部操作 |
| **LIBERO 评测脚本** | `LIBERO/scripts/run_eval.py` | 360 | WebSocket 评测客户端 |
| **OpenVLA 适配器** | `openvla/vla-scripts/policy_server.py` | 432 | obs 格式转换 + VLA 推理 + action 后处理 |

### 2.2 关键设计决策

| 决策 | 为什么 |
|------|--------|
| 每个模型独立 venv | openvla 要 transformers 4.40, octo 要 JAX 0.4.30, 放一起必冲突 |
| WebSocket 而非 HTTP | 双向流式, 一次连接持续整个 episode, 不用每步握手 |
| msgpack 而非 JSON | numpy array 直接二进制, 256x256 图 ~196KB (JSON base64 要 ~260KB) |
| SLURM sbatch 而非 srun | sbatch 提交后立即返回, Agent 不用等 2 小时 |
| LLM 选 skill 而非 hardcode | 遇到新错误能自主推理调整, if-else 覆盖不了 |
| CUDA_VISIBLE_DEVICES 作 prefix | 只影响 server 进程, 不影响同 job 里的 eval 进程 |
| MUJOCO_GL=egl 只在 eval 前设 | 全局设置会让 server 进程的 TensorFlow 尝试初始化 EGL, 导致 crash |

---

## 3. 核心代码清单

### 3.1 agent.py — 大脑 (578 行)

**做 4 件事：**

**A. 加载技能 (57-84 行)** — 扫描 `skills/*/SKILL.md`，用正则切出 YAML frontmatter，解析参数定义和命令模板。

**B. 自动发现 (87-173 行)** — 扫描 `*/policy_server.yaml`，动态生成 deploy skill。YAML 里的 `entry_point`、`arguments`、`setup` 被拼成一段 SSH 部署脚本。

**C. ReAct 循环 (428-514 行)** — 最多 15 轮：调 LLM → 解析 tool_call → `execute_skill()` → 把 stdout 喂回 LLM。

**D. JSON 容错 (299-320 行)** — Qwen 偶尔输出截断 JSON，自动补括号/正则提取/包装成 string。修复失败则把错误发回 LLM 让它重试。

### 3.2 policy_websocket — 通信层 (4 个文件, 280 行)

```python
# base_policy.py (13 行) — 所有模型只需实现这 2 个方法
class BasePolicy(ABC):
    def infer(self, obs: Dict) -> Dict: ...  # 收观测, 返动作
    def reset(self): ...                      # episode 开始时重置

# websocket_server.py (128 行) — async 服务端
# 握手时发 metadata → 循环 { recv(obs) → infer → send(action) }
# 特性: SO_REUSEADDR, /healthz 端点, SIGTERM 优雅关闭

# websocket_client.py (85 行) — 同步客户端
# connect → recv metadata → 循环 { send(obs) → recv(action) }
# 特性: 自动重连, 超时处理

# msgpack_numpy.py (54 行) — numpy 二进制序列化
# ndarray → {__ndarray__: True, data: 原始字节, dtype: "<f8", shape: (256,256,3)}
```

### 3.3 run_benchmark.py — 评测引擎 (428 行)

三种模式：
1. **SLURM Submit (默认)** — 生成 sbatch 脚本 (server + healthcheck + eval)，`sbatch` 提交，立即返回 job ID
2. **Existing Server** — 连已有 server，只跑评测
3. **Local** — 本地起 server + 跑评测

### 3.4 fix_deps.py — 依赖自修复 (~500 行)

```
迭代循环:
  for round in range(max_retries):
      failures = test_imports(python, modules)    # 在 venv 里试 import
      for failure in failures:
          match_pattern(failure)                   # 15+ 已知错误模式
          apply_fix(failure)                       # uv pip install ...
      if no_failures: break
```

15+ 错误模式: libGL.so → headless opencv, numpy 2.x → pin <2, mujoco crash → pin 2.3.7 ...
50+ 包名映射: cv2 → opencv-python-headless, sklearn → scikit-learn, yaml → pyyaml ...

### 3.5 各模型的 policy_server.py — 适配器

| 模型 | 文件 | 来源 |
|------|------|------|
| OpenVLA | `openvla/vla-scripts/policy_server.py` | 手写 (432 行, obs 格式检测 + 翻转 + VLA 推理 + gripper 后处理) |
| Octo | `octo/policy_server.py` | 手写 |
| OpenPI | `openpi/scripts/policy_server.py` | 手写 |
| Diffusion Policy | `diffusion_policy/policy_server.py` | 手写 |
| BESO | `beso/policy_server.py` | **Agent 自动生成** |
| VQ-BeT | `vq_bet/policy_server.py` | **Agent 自动生成** |
| 3D-DP | `3D-Diffusion-Policy/policy_server.py` | **Agent 自动生成** |

---

## 4. Skill 为什么有效 — LLM API 如何操控一切

### 4.1 核心机制: SKILL.md → LLM Tool → Shell 执行

整个系统的关键创新在于：**把 shell 命令包装成 LLM 能理解和调用的函数**。

一个 SKILL.md 文件有两个身份：
- **给 LLM 看**: YAML frontmatter 的 `name`、`description`、`parameters` 被转成 JSON Schema，作为 LLM API 的 tool definition
- **给机器执行**: `command_template` 是一段参数化的 bash 脚本，`execute_skill()` 替换 `{param}` 后 `subprocess.run`

```
SKILL.md (人写的声明)
     │
     ├─→ skill_to_claude_tool() ─→ JSON Schema ─→ LLM API 的 tools 参数
     │                                              LLM 看到: "我有 deploy_openvla 函数"
     │
     └─→ execute_skill() ─→ 参数替换 ─→ subprocess.run("bash -c", cmd)
                                          真正执行: ssh cn06 "nohup python3 ..."
```

### 4.2 具体的 API 调用过程

**发给 LLM 的请求** (以 Qwen 为例):

```json
POST https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
{
  "model": "qwen-plus",
  "messages": [
    {
      "role": "system",
      "content": "You are Robot Ops Agent... (SOUL.md 93行，告诉 LLM 它管理 HPC 集群，有 19 个 skill)"
    },
    {
      "role": "user",
      "content": "用openvla跑LIBERO-spatial"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "check_cluster_status",
        "description": "Check GPU cluster status — running SLURM jobs, node availability...",
        "parameters": {
          "type": "object",
          "properties": {
            "node": {"type": "string", "description": "Optional: specific compute node..."}
          },
          "required": []
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "run_benchmark",
        "description": "Run a complete benchmark evaluation...",
        "parameters": {
          "type": "object",
          "properties": {
            "policy": {"type": "string", "description": "Policy name (e.g. openvla)"},
            "benchmark": {"type": "string", "description": "libero_spatial, libero_object..."},
            "submit": {"type": "string", "description": "If 'true', submit as SLURM job (default: true)"}
          },
          "required": ["policy", "checkpoint", "benchmark"]
        }
      }
    }
  ]
}
```

**LLM 返回** (标准 OpenAI tool_calls 格式):

```json
{
  "choices": [{
    "message": {
      "content": "我先检查集群状态。",
      "tool_calls": [{
        "id": "call_abc123",
        "function": {
          "name": "check_cluster_status",
          "arguments": "{}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

**agent.py 的处理** (第 484-513 行):

```python
# 1. 解析 tool_call
tc = {"id": "call_abc123", "name": "check_cluster_status", "arguments": {}}

# 2. 找到对应 skill
skill = skill_map["check_cluster_status"]

# 3. execute_skill: 参数替换 → subprocess.run
#    command_template 里的 {node} 被替换为 "" (空)
#    → bash 执行: squeue -u yd66byne; sinfo -N ...
result = "=== SLURM Jobs ===\ncn26 idle...\n=== Node Availability ===\n..."

# 4. 把结果作为 tool_result 发回 LLM
messages.append({"role": "user", "content": [{
    "type": "tool_result",
    "tool_use_id": "call_abc123",
    "content": result  # squeue/sinfo 的 stdout 文本
}]})

# 5. 下一轮: LLM 看到集群状态，选择下一个 tool_call
```

### 4.3 为什么这比写脚本更强

**脚本方式** (if-else):
```bash
# 固定流程, 遇到任何意外就挂
sbatch eval_openvla.sh cn19
# cn19 满了? 脚本挂了. 端口冲突? 脚本挂了. 依赖缺失? 脚本挂了.
```

**Agent 方式** (ReAct):
```
Turn 1: check_cluster_status()
        → "cn19 mixed (被占用), cn26 idle"
Turn 2: LLM 推理: "cn26 空闲, 用 cn26"
        → run_benchmark(node="cn26", ...)
        → "Port 18800 already in use"
Turn 3: LLM 推理: "端口冲突, 先停掉旧 server"
        → stop_policy_server(node="cn26", port=18800)
Turn 4: → run_benchmark(node="cn26", ...)
        → "Submitted batch job 109598"
```

**区别在于**: LLM 读取每一步的 stdout/stderr，然后**推理**下一步。它能处理任何文本形式的错误信息，不需要你提前预见所有可能的故障。

### 4.4 Skill 生效的 3 个必要条件

**条件 1: description 要写对** — LLM 根据 description 决定什么时候用这个 skill。如果 `run_benchmark` 的 description 写得模糊，LLM 可能会选错工具（比如手动起 server 再跑 eval，而不是一步到位）。

**条件 2: parameters 要精确** — LLM 需要知道每个参数的含义和默认值。如果 `submit` 参数的 description 没说 "default: true"，LLM 可能忘记传这个参数。

**条件 3: command_template 的 stdout 要有意义** — 执行结果是纯文本，LLM 要能从中判断成功/失败。如果脚本只输出 "done" 而不输出具体信息，LLM 无法做后续决策。

### 4.5 SOUL.md 的作用

SOUL.md 不是可有可无的文档——它是 LLM 的 **system prompt**，直接决定 LLM 的行为质量。

关键内容：
- **3 个完整示例**: 告诉 LLM "一句话 benchmark" / "一句话集成" / "快速部署" 分别该调哪些 skill、什么顺序
- **规则 #1**: "Always check cluster status before deploying" — 不写这条，LLM 可能直接部署到一个满载节点
- **规则 #8**: "Prefer run_benchmark over manual server+eval steps" — 不写这条，LLM 可能拆成 deploy_openvla + test_connection + 手动跑 eval

**这些规则是通过实际测试 Agent 后总结出来的**——每发现一次 LLM 选错 skill 的情况，就加一条规则纠正。

---

## 5. 完整 Pipeline：从用户打字到 94% 成功率

### 场景: `python agentic/robot_agent/agent.py "用openvla跑LIBERO-spatial"`

### 阶段 0: 启动 (login node, <1 秒)

```
agent.py:main()
├─ load_skills("agentic/robot_agent/skills/")
│  → 扫描 16 个子目录, 每个读 SKILL.md, 正则切 YAML → 16 个 skill dict
│
├─ discover_policy_servers("agentRobot/")
│  → 扫描 5 个 policy_server.yaml → 跳过已有的 → 新增 3 个 deploy skill
│
├─ skill_to_claude_tool() × 19
│  → 每个 skill 的 parameters → JSON Schema → LLM tool definition
│
└─ 准备:
   tools = [19 个 tool JSON]
   skill_map = {"run_benchmark": skill_obj, "check_cluster_status": skill_obj, ...}
   messages = [{"role": "user", "content": "用openvla跑LIBERO-spatial"}]
```

### 阶段 1: Turn 1 — LLM 选择 check_cluster_status (~2 秒)

```
→ HTTP POST to LLM API
  system = SOUL.md (93 行)
  user = "用openvla跑LIBERO-spatial"
  tools = [19 个 tool definition]

← LLM 返回:
  text = "我先检查集群状态"
  tool_calls = [check_cluster_status()]

→ execute_skill("check_cluster_status", {})
  cmd = "squeue -u yd66byne ...; sinfo -N ..."
  subprocess.run(["bash", "-c", cmd])

← stdout:
  "=== SLURM Jobs === ... cn06 RUNNING ...
   === Node Availability === ... cn26 idle ..."

→ 结果发回 LLM (作为 tool_result)
```

### 阶段 2: Turn 2 — LLM 选择 run_benchmark (~2 秒)

```
→ HTTP POST to LLM API
  messages = [user原文, assistant思考+tool_call, user tool_result]

← LLM 返回:
  tool_calls = [run_benchmark(
    policy="openvla",
    checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
    benchmark="libero_spatial",
    submit="true"
  )]

→ execute_skill("run_benchmark", {...})
  填默认值: port=18800, gpu_id=0
  参数替换: {policy} → "openvla", {benchmark} → "libero_spatial", ...
  subprocess.run("python3 run_benchmark.py --policy openvla ... --submit")
```

### 阶段 3: run_benchmark.py 内部 (login node, ~3 秒)

```
run_benchmark.py:main()
├─ resolve_policy("openvla") → POLICY_CONFIGS["openvla"]
│  → server_python, server_script, server_args, env_vars, ...
│
├─ args.submit = True → submit_as_slurm_job()
│
├─ 生成 sbatch 脚本:
│  #!/bin/bash
│  #SBATCH --gres=gpu:1 --time=02:00:00
│
│  # Step 1: 启动 policy server (后台)
│  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.../agentic/policy_websocket/src:.../openvla \
│    .../openvla/.venv/bin/python3 .../policy_server.py \
│    --pretrained_checkpoint moojink/openvla-7b-oft-... --port 18800 &
│
│  # Step 2: 健康检查 (每 10 秒 curl /healthz)
│  for i in $(seq 1 60); do curl localhost:18800/healthz && break; sleep 10; done
│
│  # Step 3: 跑评测
│  export MUJOCO_GL=egl
│  PYTHONPATH=.../agentic/policy_websocket/src:.../LIBERO \
│    .../LIBERO/.venv/bin/python3 run_eval.py \
│    --policy_server_addr localhost:18800 --policy openvla-oft ...
│
│  kill $SERVER_PID
│
├─ subprocess.run(["sbatch", script]) → "Submitted batch job 109598"
└─ 返回 job ID 给 agent
```

### 阶段 4: SLURM Job 执行 (GPU 节点 cn26, ~11 分钟)

#### 进程 A: OpenVLA Server

```
policy_server.py:main()
├─ OpenVLAPolicy.__init__()
│  ├─ get_vla() → from_pretrained("moojink/openvla-7b-oft-...")
│  │   → 加载 4 个 shard (~14GB) 到 GPU 0 (~30 秒)
│  ├─ get_processor() → tokenizer + image processor
│  ├─ get_action_head() → L1 回归头
│  └─ get_proprio_projector() → 本体感觉映射
│
├─ WebsocketPolicyServer(policy, port=18800, metadata={action_dim: 7, ...})
│  ├─ socket.bind("0.0.0.0", 18800)
│  ├─ /healthz → HTTP 200 OK (curl 检测通过)
│  └─ 等待 WebSocket 连接...
│
└─ 连接到来后, _handler 循环:
   while True:
     obs = msgpack.unpackb(recv())     # ~400KB: 两张图 + 传感器
     action = policy.infer(obs)         # ~59ms GPU forward pass
     send(msgpack.pack(action))         # ~120 bytes: 7D action
```

#### 进程 B: LIBERO Eval

```
run_eval.py:main()
├─ WebsocketClientPolicy(localhost:18800)
│  ├─ websocket.connect("ws://localhost:18800")
│  └─ recv metadata → {"policy_name": "OpenVLAPolicy", "action_dim": 7}
│
├─ 验证: action_dim=7 == expected(cartesian_pose)=7 ✓
│
├─ task_suite = benchmark["libero_spatial"]() → 10 个 task
│
└─ 评测循环: 10 tasks × 5 episodes = 50 episodes
   for task_id in range(10):
     env = OffScreenRenderEnv(task, controller="OSC_POSE", 256×256)
     for episode in range(5):
       env.reset() → obs
       for t in range(220):
         result = policy.infer({**obs, "task_description": "..."})
         action = result["actions"]     # (7,) float64
         obs, reward, done, _ = env.step(action)
         if done: success! break
```

### 阶段 5: 最终结果

```
Overall success rate:  0.9400 (94.0%)
  [0] pick up the black bowl between plate and ramekin:   100%
  [1] pick up the black bowl next to the ramekin:         100%
  [2] pick up the black bowl from table center:           100%
  [3] pick up the black bowl on the cookie box:           100%
  [4] pick up the black bowl in top drawer:               100%
  [5] pick up the black bowl on the ramekin:              100%
  [6] pick up the black bowl next to cookie box:          100%
  [7] pick up the black bowl on the stove:                 80%
  [8] pick up the black bowl next to the plate:            80%
  [9] pick up the black bowl on the wooden cabinet:        80%
```

---

## 6. 通信协议：WebSocket + msgpack

### 为什么选 WebSocket + msgpack

| 方案 | 问题 |
|------|------|
| HTTP REST + JSON | 每步都要握手 (3-way handshake), 图片 base64 编码膨胀 33%, 延迟高 |
| gRPC + protobuf | 需要 .proto 定义, 每换一个模型都要改 schema, 太重 |
| pickle over TCP | 不安全 (arbitrary code execution), 不跨语言 |
| **WebSocket + msgpack** | 一次连接, 二进制帧, schema-free, 安全, 快 |

### Wire 格式

```
每一步 (每 ~60ms):

Client → Server (~400KB):
┌─────────────────────────────────────────────────────┐
│ msgpack({                                           │
│   "agentview_image": {                              │
│     __ndarray__: true,                              │
│     data: [196608 bytes],  ← 256×256×3 原始像素     │
│     dtype: "|u1",          ← uint8                  │
│     shape: [256, 256, 3]                            │
│   },                                                │
│   "robot0_eye_in_hand_image": { 同上 196608 bytes },│
│   "robot0_eef_pos": { data: [24 bytes], shape: [3] }│
│   "robot0_eef_quat": { shape: [4] },               │
│   "robot0_gripper_qpos": { shape: [2] },            │
│   "task_description": "pick up the black bowl..."   │
│ })                                                  │
└─────────────────────────────────────────────────────┘

Server → Client (~120 bytes):
┌─────────────────────────────────────────────────────┐
│ msgpack({                                           │
│   "actions": {                                      │
│     __ndarray__: true,                              │
│     data: [56 bytes],      ← 7 × float64           │
│     dtype: "<f8",                                   │
│     shape: [1, 7]          ← [dx,dy,dz,dax,day,daz,gripper] │
│   },                                                │
│   "server_timing": {"infer_ms": 59.2}              │
│ })                                                  │
└─────────────────────────────────────────────────────┘
```

---

## 7. 模型推理：单步数据流

以 OpenVLA 处理一帧 LIBERO 观测为例：

```
LIBERO env.step()
│
│ 输出 obs:
│   agentview_image: uint8 (256, 256, 3)    ← 第三人称 RGB
│   robot0_eye_in_hand_image: uint8 (256, 256, 3) ← 腕部 RGB
│   robot0_eef_pos: float64 (3,)             ← 末端位置 [x,y,z]
│   robot0_eef_quat: float64 (4,)            ← 末端四元数
│   robot0_gripper_qpos: float64 (2,)        ← 夹爪开合
│
├─ WebSocket Client: msgpack.pack(obs) → 发送 ~400KB
│
│ ====== 网络传输 (localhost, <1ms) ======
│
├─ WebSocket Server: msgpack.unpackb() → numpy 重建
│
├─ remap_obs_to_openvla():
│   ├─ 检测到 "agentview_image" → LIBERO 格式
│   ├─ np.fliplr(np.flipud(image))          ← LIBERO 渲染是上下颠倒的, 旋转 180°
│   ├─ state = concat([eef_pos(3), quat2axisangle(quat)(3), gripper(2)]) → (8,)
│   └─ 输出: {full_image, wrist_image, state, task_description}
│
├─ get_vla_action():                          ← ~59ms on H100
│   ├─ processor: resize 224×224, normalize, tokenize 文本
│   ├─ VLA forward: 7B transformer (GPU)
│   │   输入: 两张图 + 文本 + 本体感觉 → 隐状态
│   ├─ action_head: 隐状态 → L1 回归 → (7,) raw action
│   └─ 输出: [dx, dy, dz, dax, day, daz, gripper_raw]
│
├─ postprocess_action_for_env():
│   ├─ normalize_gripper: [0,1] → [-1,+1], 二值化
│   └─ invert_gripper: 翻转符号 (LIBERO: -1=开, +1=关)
│
├─ WebSocket Server: msgpack.pack({actions, server_timing}) → 发送 ~120 bytes
│
│ ====== 网络传输 ======
│
├─ WebSocket Client: unpackb → action (7,) float64
│
├─ pad_action_for_env(): 7D → 7D (cartesian_pose 不需要 pad)
│
└─ env.step(action)
   ├─ MuJoCo 物理仿真: 移动机械臂, 碰撞检测, 重力
   ├─ EGL 离屏渲染: 生成下一帧 256×256 图像
   └─ 输出: new_obs, reward, done, info
```

---

## 8. 验证过的完整流程

### 流程 A: 一句话 Benchmark (已验证, 94%)

```
用户: "用openvla跑LIBERO-spatial"
Agent:
  Turn 1: check_cluster_status() → "cn26 idle"
  Turn 2: run_benchmark(policy="openvla", benchmark="libero_spatial", submit="true")
          → "Submitted batch job 109598"
结果: SLURM job 在 cn26 H100 上跑 11 分钟 → 94.0% (47/50)
```

### 流程 B: 一句话集成新仓库 (已验证, BESO)

```
用户: "帮我集成 https://github.com/intuitive-robots/beso"
Agent (5 步全自动):
  1. analyze_repo(repo_url=...) → 克隆, 找到 BesoAgent, 检测 Hydra+PyTorch
  2. setup_env(repo_path=.../beso) → 创建 venv, 装 torch cu121 + beso
  3. fix_deps(repo_path=.../beso) → 自动装 msgpack, websockets, pin numpy<2
  4. wrap_policy(model_class="BesoAgent", framework="torch") → 生成 115 行 policy_server.py
  5. create_deploy_skill(skill_name="deploy_beso") → 生成部署 skill
结果: deploy_beso skill 立即可用, 0 人工干预
```

### 流程 C: 智能错误恢复 (已验证, 15 轮 ReAct)

```
用户: "用openvla在LIBERO-spatial上评测，帮我找个可用GPU节点"
Agent 问题解决 (15 轮):
  1. 找到 cn30 空闲 → SSH 被拒 (没有活跃 SLURM job)
  2. 换 cn19 → SSH 被拒 (同样原因)
  3. 换 cn06 (有活跃 job, SSH 可用) → 部署成功
  4. 测试连接 → 超时 (模型还在加载)
  5. 重试测试 → 连接成功, 747ms 推理
  6. 跑 benchmark → 端口冲突 (server 已占用)
  7. 停掉 server → 重试 → login node 无 GPU
  8. 诊断 → 正确识别每个失败原因并调整策略
```

---

## 9. 已集成模型

| 模型 | 类型 | 框架 | 推理速度 | 状态 |
|------|------|------|----------|------|
| OpenVLA-OFT 7B | Vision-Language-Action | PyTorch | 59ms/step (H100) | **94% on LIBERO-spatial** |
| Octo | Transformer policy | JAX | 184ms | 已验证 |
| OpenPI (pi0) | Flow-based VLA | JAX | 5ms (post-JIT) | 已验证 |
| Diffusion Policy | DDPM denoiser | PyTorch | adapter ready | 已集成 |
| Robomimic | Behavior cloning | PyTorch | adapter ready | 已集成 |
| VQ-BeT | Quantized transformer | PyTorch | adapter ready | 已集成 |
| BESO | Score-based diffusion | PyTorch | adapter ready | Agent 自动集成 |
| 3D Diffusion Policy | Point cloud diffusion | PyTorch | 需 pytorch3d | 部分集成 |

---

## 10. 目录结构

```
agentRobot/
│
├── agentic/                      ← 我们的工作 (本目录)
│   ├── OVERVIEW.md               ← 本文档
│   ├── robot_agent/              ← Agent 大脑
│   │   ├── agent.py              ← ReAct 循环 (578 行)
│   │   ├── SOUL.md               ← System prompt (93 行)
│   │   └── skills/               ← 19 个 skill
│   │       ├── analyze_repo/     ← 克隆+分析 GitHub 仓库
│   │       ├── wrap_policy/      ← 自动生成 policy_server.py 适配器
│   │       │   └── generate_smart.py  ← 3 种模式检测
│   │       ├── setup_env/        ← 创建 venv + 装依赖
│   │       ├── fix_deps/         ← 依赖自动修复
│   │       │   └── fix_deps.py   ← 15+ 错误模式, 50+ 包名映射
│   │       ├── run_benchmark/    ← 端到端评测
│   │       │   └── run_benchmark.py  ← SLURM/existing/local 三种模式
│   │       ├── build_container/  ← Apptainer 容器
│   │       ├── create_deploy_skill/  ← 自扩展 skill 生成器
│   │       ├── check_cluster_status/ ← SLURM 集群查询
│   │       ├── deploy_openvla/   ← OpenVLA 部署
│   │       ├── deploy_octo/      ← Octo 部署
│   │       ├── deploy_beso/      ← BESO 部署 (Agent 自动生成)
│   │       ├── deploy_diffusion_policy/
│   │       ├── deploy_robomimic/
│   │       ├── test_policy_connection/ ← WebSocket 连接测试
│   │       ├── stop_policy_server/     ← 停止 server
│   │       └── write_file/       ← 文件写入
│   │
│   ├── policy_websocket/         ← 通信层
│   │   └── src/policy_websocket/
│   │       ├── base_policy.py    ← 13 行抽象接口
│   │       ├── websocket_server.py ← 128 行 async 服务端
│   │       ├── websocket_client.py ← 85 行同步客户端
│   │       └── msgpack_numpy.py  ← 54 行 numpy 序列化
│   │
│   ├── scripts/                  ← 手动评测脚本
│   │   ├── eval_openvla_libero.sh
│   │   └── eval_debug.sh
│   │
│   └── containers/               ← Apptainer 容器定义
│       ├── policy_base.def
│       └── diffusion_policy.def
│
├── openvla/                      ← OpenVLA-OFT 7B (PyTorch)
│   ├── vla-scripts/policy_server.py  ← 432 行适配器 (手写)
│   └── policy_server.yaml        ← 自动发现元数据
├── LIBERO/                       ← 130+ 操作任务 benchmark
│   └── scripts/run_eval.py       ← 360 行 WebSocket 评测客户端 (手写)
├── octo/                         ← Octo (JAX)
├── openpi/                       ← pi0 VLA (JAX)
├── diffusion_policy/             ← Diffusion Policy (PyTorch)
├── beso/                         ← BESO (Agent 自动集成)
├── vq_bet/                       ← VQ-BeT (自动集成)
├── 3D-Diffusion-Policy/          ← DP3 (部分集成)
├── droid_policy_learning/        ← DROID 数据集训练
├── logs/                         ← SLURM 日志 + 评测结果
└── README.md
```

---

## 使用方法

```bash
# 设置 API Key (三选一)
export DASHSCOPE_API_KEY=sk-xxx    # Qwen
export ANTHROPIC_API_KEY=sk-xxx    # Claude
export OPENAI_API_KEY=sk-xxx       # GPT

# 交互模式
python agentic/robot_agent/agent.py
# You> 用openvla跑LIBERO-spatial
# You> 集成 https://github.com/xxx/new-model
# You> 检查集群哪些节点有空闲GPU

# 单条命令模式
python agentic/robot_agent/agent.py "在cn19上部署openvla并测试连接"

# 直接调用 skill
python agentic/robot_agent/skills/run_benchmark/run_benchmark.py \
  --policy openvla --checkpoint moojink/openvla-7b-oft-finetuned-libero-spatial \
  --benchmark libero_spatial --num_trials 5 --submit
```
