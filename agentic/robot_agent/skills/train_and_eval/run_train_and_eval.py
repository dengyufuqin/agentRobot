#!/usr/bin/env python3
"""
run_train_and_eval.py — Full pipeline: finetune → deploy → evaluate in one SLURM job.

Generates a SLURM script with three phases:
  Phase 1: Finetune the model (lerobot-train or torchrun finetune.py)
  Phase 2: Start policy server with the new checkpoint
  Phase 3: Run benchmark evaluation
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

AGENT_ROOT = Path(os.environ.get("AGENTROBOT_ROOT", Path(__file__).resolve().parents[4]))
CHECKPOINT_DIR = AGENT_ROOT / "checkpoints"
LOGS_DIR = AGENT_ROOT / "logs"
EGL_EXCLUDE_NODES = os.environ.get("EGL_EXCLUDE_NODES", "cn19,cn23")

# ============================================================================
#  Config tables
# ============================================================================

# LeRobot model configs
LEROBOT_MODELS = {
    "pi0": {
        "base_checkpoint": "lerobot/pi0_libero_finetuned",
        "server_script": str(AGENT_ROOT / "lerobot" / "policy_server.py"),
        "server_python": str(AGENT_ROOT / "lerobot" / ".venv" / "bin" / "python3"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "lerobot"),
        ],
        "train_python": str(AGENT_ROOT / "lerobot" / ".venv" / "bin" / "python3"),
        "train_script": str(AGENT_ROOT / "lerobot" / "src" / "lerobot" / "scripts" / "lerobot_train.py"),
        "train_pythonpath": [
            str(AGENT_ROOT / "lerobot" / "src"),
            str(AGENT_ROOT / "lerobot"),
        ],
        "policy_flag": "lerobot",
        "arm_controller": "cartesian_pose",
    },
    "pi0.5": {
        "base_checkpoint": "lerobot/pi05_libero_finetuned",
        "server_script": str(AGENT_ROOT / "lerobot" / "policy_server.py"),
        "server_python": str(AGENT_ROOT / "lerobot" / ".venv" / "bin" / "python3"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "lerobot"),
        ],
        "train_python": str(AGENT_ROOT / "lerobot" / ".venv" / "bin" / "python3"),
        "train_script": str(AGENT_ROOT / "lerobot" / "src" / "lerobot" / "scripts" / "lerobot_train.py"),
        "train_pythonpath": [
            str(AGENT_ROOT / "lerobot" / "src"),
            str(AGENT_ROOT / "lerobot"),
        ],
        "policy_flag": "lerobot",
        "arm_controller": "cartesian_pose",
    },
    "openvla": {
        "base_checkpoint": "openvla/openvla-7b",
        "server_script": str(AGENT_ROOT / "openvla" / "vla-scripts" / "policy_server.py"),
        "server_python": str(AGENT_ROOT / "openvla" / ".venv" / "bin" / "python3"),
        "server_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "openvla"),
        ],
        "train_python": str(AGENT_ROOT / "openvla" / ".venv" / "bin" / "python3"),
        "train_script": str(AGENT_ROOT / "openvla" / "vla-scripts" / "finetune.py"),
        "train_pythonpath": [str(AGENT_ROOT / "openvla")],
        "policy_flag": "openvla-oft",
        "arm_controller": "cartesian_pose",
    },
}

# Dataset mapping
LEROBOT_DATASETS = {
    "libero_spatial": "HuggingFaceVLA/libero",
    "libero_object": "HuggingFaceVLA/libero",
    "libero_goal": "HuggingFaceVLA/libero",
    "libero_10": "HuggingFaceVLA/libero",
}

# Eval client configs
EVAL_CLIENTS = {
    "libero": {
        "eval_python": str(AGENT_ROOT / "LIBERO" / ".venv" / "bin" / "python3"),
        "eval_script": str(AGENT_ROOT / "LIBERO" / "scripts" / "run_eval.py"),
        "eval_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "LIBERO"),
        ],
        "eval_env": "export MUJOCO_GL=egl",
    },
    "robocasa": {
        "eval_python": str(AGENT_ROOT / "robocasa" / ".venv" / "bin" / "python3"),
        "eval_script": str(AGENT_ROOT / "robocasa" / "scripts" / "run_eval.py"),
        "eval_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "robocasa"),
        ],
        "eval_env": "export MUJOCO_GL=egl",
    },
    "maniskill": {
        "eval_python": str(AGENT_ROOT / "ManiSkill" / ".venv" / "bin" / "python3"),
        "eval_script": str(AGENT_ROOT / "ManiSkill" / "scripts" / "run_eval.py"),
        "eval_pythonpath": [
            str(AGENT_ROOT / "agentic" / "policy_websocket" / "src"),
            str(AGENT_ROOT / "ManiSkill"),
        ],
        "eval_env": "",
    },
}

# Benchmark → eval_client mapping
BENCHMARK_TO_CLIENT = {
    "libero_spatial": ("libero", "libero_spatial"),
    "libero_object": ("libero", "libero_object"),
    "libero_goal": ("libero", "libero_goal"),
    "libero_10": ("libero", "libero_10"),
}


def resolve_checkpoint_path(output_dir):
    """Predict where lerobot-train will save the final checkpoint."""
    # LeRobot saves as: output_dir/checkpoints/<step>/pretrained_model/
    # We use a glob at runtime in the SLURM script
    return f"{output_dir}/checkpoints/last/pretrained_model"


def build_train_block(args, model_cfg, output_dir):
    """Generate the training phase bash block."""
    if args.policy == "openvla":
        # OpenVLA uses torchrun
        return _build_openvla_train(args, model_cfg, output_dir)
    else:
        # LeRobot training
        return _build_lerobot_train(args, model_cfg, output_dir)


def _build_lerobot_train(args, model_cfg, output_dir):
    """LeRobot training block."""
    python = model_cfg["train_python"]
    script = model_cfg["train_script"]
    pp = ":".join(model_cfg["train_pythonpath"])
    dataset = LEROBOT_DATASETS.get(args.benchmark, args.benchmark)
    base_ckpt = model_cfg["base_checkpoint"]
    save_freq = max(args.train_steps // 5, 1000)

    return f"""echo "=========================================="
echo "  PHASE 1: FINETUNE ({args.policy} on {args.benchmark})"
echo "=========================================="
echo "Base checkpoint: {base_ckpt}"
echo "Dataset: {dataset}"
echo "Steps: {args.train_steps}, BS: {args.batch_size}, LR: {args.learning_rate}"
echo ""

PYTHONPATH={pp} {python} {script} \\
    --dataset.repo_id={dataset} \\
    --policy.path={base_ckpt} \\
    --policy.push_to_hub=false \\
    --output_dir={output_dir} \\
    --batch_size={args.batch_size} \\
    --steps={args.train_steps} \\
    --save_freq={save_freq} \\
    --eval_freq={save_freq} \\
    --log_freq=100 \\
    --num_workers=4 \\
    --seed=1000 \\
    --optimizer.lr={args.learning_rate} \\
    --use_policy_training_preset=true
TRAIN_EXIT=$?

echo ""
echo "=== Training exit: $TRAIN_EXIT ==="
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "TRAINING FAILED — aborting pipeline"
    exit $TRAIN_EXIT
fi

# Find the latest checkpoint
CKPT_DIR=$(ls -td {output_dir}/checkpoints/*/pretrained_model 2>/dev/null | head -1)
if [ -z "$CKPT_DIR" ]; then
    echo "ERROR: No checkpoint found after training"
    exit 1
fi
echo "Checkpoint: $CKPT_DIR"
"""


def _build_openvla_train(args, model_cfg, output_dir):
    """OpenVLA training block."""
    script = model_cfg["train_script"]
    pp = ":".join(model_cfg["train_pythonpath"])
    base_ckpt = model_cfg["base_checkpoint"]

    openvla_ds_map = {
        "libero_spatial": "libero_spatial_no_noops",
        "libero_object": "libero_object_no_noops",
        "libero_goal": "libero_goal_no_noops",
        "libero_10": "libero_10_no_noops",
    }
    dataset_name = openvla_ds_map.get(args.benchmark, args.benchmark)
    data_root = "/mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/rlds"
    save_freq = max(args.train_steps // 5, 1000)

    return f"""echo "=========================================="
echo "  PHASE 1: FINETUNE (openvla on {args.benchmark})"
echo "=========================================="
echo "Base: {base_ckpt}"
echo "Dataset: {dataset_name} (RLDS)"
echo "Steps: {args.train_steps}, BS: {args.batch_size}, LR: {args.learning_rate}"
echo ""

PYTHONPATH={pp} torchrun --standalone --nnodes=1 --nproc-per-node=1 \\
    {script} \\
    --vla_path={base_ckpt} \\
    --data_root_dir={data_root} \\
    --dataset_name={dataset_name} \\
    --run_root_dir={output_dir} \\
    --batch_size={args.batch_size} \\
    --max_steps={args.train_steps} \\
    --learning_rate={args.learning_rate} \\
    --lora_rank=32 \\
    --save_freq={save_freq} \\
    --use_l1_regression=True \\
    --image_aug=True
TRAIN_EXIT=$?

echo ""
echo "=== Training exit: $TRAIN_EXIT ==="
if [ $TRAIN_EXIT -ne 0 ]; then
    echo "TRAINING FAILED — aborting pipeline"
    exit $TRAIN_EXIT
fi

CKPT_DIR=$(ls -td {output_dir}/*/merged* 2>/dev/null | head -1)
if [ -z "$CKPT_DIR" ]; then
    CKPT_DIR=$(ls -td {output_dir}/*/checkpoint* 2>/dev/null | head -1)
fi
echo "Checkpoint: $CKPT_DIR"
"""


def build_eval_block(args, model_cfg, ckpt_var="$CKPT_DIR"):
    """Generate the eval phase bash block (server + eval client)."""
    bench_key = args.benchmark
    if bench_key not in BENCHMARK_TO_CLIENT:
        # Try stripping prefix for robocasa:TaskName format
        if ":" in bench_key:
            platform = bench_key.split(":")[0]
        elif bench_key.startswith("libero_"):
            platform = "libero"
        else:
            platform = bench_key
        task_id = bench_key.split(":")[-1] if ":" in bench_key else bench_key
        client_name = platform
    else:
        client_name, task_id = BENCHMARK_TO_CLIENT[bench_key]

    if client_name not in EVAL_CLIENTS:
        return f'echo "ERROR: No eval client for {client_name}"; exit 1'

    client = EVAL_CLIENTS[client_name]
    port = 18800

    # Server setup
    server_python = model_cfg["server_python"]
    server_script = model_cfg["server_script"]
    pp_server = ":".join(model_cfg["server_pythonpath"])
    policy_flag = model_cfg["policy_flag"]
    arm_controller = model_cfg["arm_controller"]

    # Server args depend on policy type
    if args.policy == "openvla":
        server_args = f"--pretrained_checkpoint {ckpt_var} --port {port} --execute_steps 1"
    else:
        server_args = f"--checkpoint {ckpt_var} --port {port}"

    # Eval setup
    eval_python = client["eval_python"]
    eval_script = client["eval_script"]
    pp_eval = ":".join(client["eval_pythonpath"])
    eval_env = client.get("eval_env", "")

    # Build eval args based on client type
    if client_name == "libero":
        eval_args = f"""--policy_server_addr localhost:{port} \\
  --policy {policy_flag} \\
  --task_suite_name {task_id} \\
  --num_trials_per_task {args.num_eval_trials} \\
  --arm_controller {arm_controller} \\
  --log_dir {AGENT_ROOT}/logs/eval_results \\
  --skip_preflight \\
  --no_save_video"""
    elif client_name == "robocasa":
        eval_args = f"""--policy_server_addr localhost:{port} \\
  --policy {policy_flag} \\
  --task_name {task_id} \\
  --num_trials {args.num_eval_trials} \\
  --arm_controller {arm_controller} \\
  --log_dir {AGENT_ROOT}/logs/eval_results \\
  --no_save_video"""
    elif client_name == "maniskill":
        eval_args = f"""--policy_server_addr localhost:{port} \\
  --policy {policy_flag} \\
  --env_id {task_id} \\
  --num_trials {args.num_eval_trials} \\
  --log_dir {AGENT_ROOT}/logs/eval_results \\
  --no_save_video"""
    else:
        eval_args = f"--policy_server_addr localhost:{port} --num_trials {args.num_eval_trials}"

    return f"""
echo ""
echo "=========================================="
echo "  PHASE 2: DEPLOY SERVER"
echo "=========================================="
echo "Starting policy server with checkpoint: {ckpt_var}"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH={pp_server} {server_python} {server_script} {server_args} &
SERVER_PID=$!

# Wait for server to be ready (WebSocket — check TCP port)
for i in $(seq 1 60); do
  python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('localhost',{port})); s.close()" 2>/dev/null && {{ echo "Server ready after ${{i}}0s"; break; }}
  kill -0 $SERVER_PID 2>/dev/null || {{ echo "Server process died"; exit 1; }}
  sleep 10
done

echo ""
echo "=========================================="
echo "  PHASE 3: EVALUATE"
echo "=========================================="
{eval_env}
PYTHONPATH={pp_eval} {eval_python} -u {eval_script} \\
  {eval_args}
EVAL_EXIT=$?

echo ""
echo "=== Eval exit: $EVAL_EXIT ==="
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
"""


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate in one SLURM job")
    parser.add_argument("--policy", required=True, help="pi0, pi0.5, openvla")
    parser.add_argument("--benchmark", required=True, help="libero_spatial, libero_object, etc.")
    parser.add_argument("--train_steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", default="2e-5")
    parser.add_argument("--num_eval_trials", type=int, default=5)
    parser.add_argument("--node", default="")

    args = parser.parse_args()

    if args.policy not in LEROBOT_MODELS:
        print(f"ERROR: Unknown policy '{args.policy}'. Known: {list(LEROBOT_MODELS.keys())}")
        sys.exit(1)

    model_cfg = LEROBOT_MODELS[args.policy]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{args.policy}-{args.benchmark}-{ts}"
    output_dir = str(CHECKPOINT_DIR / output_name)

    # Estimate time: ~2000 steps/hr for LeRobot, + 30min for eval
    train_hours = max(1, args.train_steps // 2000)
    total_hours = min(train_hours + 1, 48)

    print(f"\n{'='*60}")
    print(f"  TRAIN AND EVAL PIPELINE")
    print(f"{'='*60}")
    print(f"  Policy:      {args.policy}")
    print(f"  Benchmark:   {args.benchmark}")
    print(f"  Train steps: {args.train_steps}")
    print(f"  Batch size:  {args.batch_size}")
    print(f"  LR:          {args.learning_rate}")
    print(f"  Eval trials: {args.num_eval_trials}")
    print(f"  Output:      {output_dir}")
    print(f"  Est. time:   ~{total_hours}h")
    print(f"{'='*60}")

    # Build SLURM script
    train_block = build_train_block(args, model_cfg, output_dir)
    eval_block = build_eval_block(args, model_cfg)

    node_line = f"#SBATCH --nodelist={args.node}" if args.node else ""
    exclude_line = f"#SBATCH --exclude={EGL_EXCLUDE_NODES}" if not args.node else ""
    job_log = str(LOGS_DIR / f"traineval-{output_name}-%j.log")

    script = f"""#!/bin/bash
#SBATCH --job-name=te-{args.policy[:5]}-{args.benchmark[:10]}
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --time={total_hours:02d}:00:00
#SBATCH --output={job_log}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
{node_line}
{exclude_line}

export HF_HOME={os.path.expanduser("~/.cache/huggingface")}
export WANDB_DISABLED=true

echo "============================================================"
echo "  TRAIN AND EVAL PIPELINE"
echo "  Policy: {args.policy} | Benchmark: {args.benchmark}"
echo "  Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "============================================================"
echo ""

{train_block}
{eval_block}

echo ""
echo "============================================================"
echo "  PIPELINE COMPLETE"
echo "  Checkpoint: $CKPT_DIR"
echo "  Eval exit: $EVAL_EXIT"
echo "============================================================"
exit $EVAL_EXIT
"""

    # Clean up empty SBATCH lines
    lines = script.split("\n")
    lines = [l for l in lines if not (l.strip().startswith("#SBATCH") and l.strip().endswith("="))]
    script = "\n".join(lines)

    # Write and submit
    script_path = str(LOGS_DIR / f"_te_{output_name}.sh")
    os.makedirs(LOGS_DIR, exist_ok=True)
    with open(script_path, "w") as f:
        f.write(script)
    os.chmod(script_path, 0o755)

    result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: sbatch failed: {result.stderr}")
        sys.exit(1)

    job_id = result.stdout.strip().split()[-1]
    print(f"\n  Submitted: {result.stdout.strip()}")
    print(f"  Script: {script_path}")
    print(f"  Log: {job_log.replace('%j', job_id)}")
    print(f"\n  Monitor: tail -f {job_log.replace('%j', job_id)}")
    print(f"  Status:  squeue -j {job_id}")


if __name__ == "__main__":
    main()
