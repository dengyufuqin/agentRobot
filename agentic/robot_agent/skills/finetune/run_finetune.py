#!/usr/bin/env python3
"""
run_finetune.py — Unified finetuning launcher for VLA models.

Supports two backends:
  - LeRobot (pi0, pi0.5, smolvla): uses lerobot-train CLI
  - OpenVLA: uses torchrun + vla-scripts/finetune.py

Submits as SLURM job for GPU training, or runs locally.
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

# Nodes to exclude (known EGL issues)
EGL_EXCLUDE_NODES = os.environ.get("EGL_EXCLUDE_NODES", "cn19,cn23")

# ============================================================================
#  Model defaults
# ============================================================================

LEROBOT_MODELS = {
    "pi0": {
        # Use LIBERO-finetuned as base (already has correct 2-camera feature mapping)
        # For truly new benchmarks, use "lerobot/pi0_base" with appropriate rename_map
        "base_checkpoint": "lerobot/pi0_libero_finetuned",
        "base_checkpoint_raw": "lerobot/pi0_base",
        "policy_type": "pi0",
    },
    "pi0.5": {
        "base_checkpoint": "lerobot/pi05_libero_finetuned",
        "base_checkpoint_raw": "lerobot/pi05_base",
        "policy_type": "pi05",
    },
    "smolvla": {
        "base_checkpoint": "lerobot/smolvla_base",
        "policy_type": "smolvla",
    },
}

OPENVLA_DEFAULTS = {
    "base_checkpoint": "openvla/openvla-7b",
    "lora_rank": 32,
    "use_l1_regression": True,
    "image_aug": True,
}

# ============================================================================
#  Dataset mapping — benchmark name → dataset config per backend
# ============================================================================

LEROBOT_DATASETS = {
    # LIBERO — use the official HuggingFaceVLA dataset (compatible with pi0/pi0.5 image keys)
    "libero": {"repo_id": "HuggingFaceVLA/libero"},
    "libero_spatial": {"repo_id": "HuggingFaceVLA/libero"},
    "libero_object": {"repo_id": "HuggingFaceVLA/libero"},
    "libero_goal": {"repo_id": "HuggingFaceVLA/libero"},
    "libero_10": {"repo_id": "HuggingFaceVLA/libero"},
    # Per-suite datasets (alternative, different image keys — need rename_map)
    "libero_spatial_image": {"repo_id": "lerobot/libero_spatial_image"},
    "libero_object_image": {"repo_id": "lerobot/libero_object_image"},
    "libero_goal_image": {"repo_id": "lerobot/libero_goal_image"},
    "libero_10_image": {"repo_id": "lerobot/libero_10_image"},
    # RoboCasa — uses HF dataset if available, else needs generation
    "robocasa": {"repo_id": "robocasa_dataset", "note": "needs data generation first"},
}

OPENVLA_DATASETS = {
    # RLDS format datasets for OpenVLA finetuning
    "libero_spatial": {"dataset_name": "libero_spatial_no_noops", "data_root": "datasets/rlds"},
    "libero_object": {"dataset_name": "libero_object_no_noops", "data_root": "datasets/rlds"},
    "libero_goal": {"dataset_name": "libero_goal_no_noops", "data_root": "datasets/rlds"},
    "libero_10": {"dataset_name": "libero_10_no_noops", "data_root": "datasets/rlds"},
}


def build_lerobot_command(args, output_dir):
    """Build lerobot-train CLI command."""
    model_cfg = LEROBOT_MODELS[args.policy]
    dataset_cfg = LEROBOT_DATASETS.get(args.benchmark)

    base_ckpt = args.base_checkpoint or model_cfg["base_checkpoint"]

    if dataset_cfg is None:
        # Try treating benchmark as a HuggingFace dataset repo_id
        repo_id = args.benchmark
    else:
        repo_id = dataset_cfg["repo_id"]

    python = str(AGENT_ROOT / "lerobot" / ".venv" / "bin" / "python3")
    train_script = str(AGENT_ROOT / "lerobot" / "src" / "lerobot" / "scripts" / "lerobot_train.py")

    cmd_parts = [
        python, train_script,
        f"--dataset.repo_id={repo_id}",
        f"--policy.path={base_ckpt}",
        f"--policy.push_to_hub=false",
        f"--output_dir={output_dir}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        f"--save_freq={max(args.steps // 5, 1000)}",
        f"--eval_freq={max(args.steps // 5, 1000)}",
        f"--log_freq=100",
        f"--num_workers=4",
        f"--seed=1000",
    ]

    if args.learning_rate:
        cmd_parts.append(f"--optimizer.lr={args.learning_rate}")

    # PEFT (LoRA) — use training preset defaults
    cmd_parts.append("--use_policy_training_preset=true")

    return cmd_parts, python


def build_openvla_command(args, output_dir):
    """Build OpenVLA finetune command (torchrun)."""
    dataset_cfg = OPENVLA_DATASETS.get(args.benchmark)
    if dataset_cfg is None:
        print(f"ERROR: No RLDS dataset mapping for benchmark '{args.benchmark}' (OpenVLA backend)")
        print(f"  Available: {list(OPENVLA_DATASETS.keys())}")
        sys.exit(1)

    base_ckpt = args.base_checkpoint or OPENVLA_DEFAULTS["base_checkpoint"]
    python = str(AGENT_ROOT / "openvla" / ".venv" / "bin" / "python3")
    finetune_script = str(AGENT_ROOT / "openvla" / "vla-scripts" / "finetune.py")

    # Check for RLDS data directory
    data_root = str(AGENT_ROOT / dataset_cfg["data_root"])
    workspace_data = "/mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/rlds"
    if os.path.isdir(workspace_data):
        data_root = workspace_data

    cmd_parts = [
        f"torchrun --standalone --nnodes=1 --nproc-per-node={args.num_gpus}",
        finetune_script,
        f"--vla_path={base_ckpt}",
        f"--data_root_dir={data_root}",
        f"--dataset_name={dataset_cfg['dataset_name']}",
        f"--run_root_dir={output_dir}",
        f"--batch_size={args.batch_size}",
        f"--max_steps={args.steps}",
        f"--learning_rate={args.learning_rate}",
        f"--lora_rank={OPENVLA_DEFAULTS['lora_rank']}",
        f"--save_freq={max(args.steps // 5, 1000)}",
        f"--use_l1_regression={'True' if OPENVLA_DEFAULTS['use_l1_regression'] else 'False'}",
        f"--image_aug={'True' if OPENVLA_DEFAULTS['image_aug'] else 'False'}",
    ]

    return cmd_parts, python


def generate_slurm_script(args, cmd_parts, python, output_dir):
    """Generate SLURM batch script for finetuning."""
    output_name = args.output_name or f"{args.policy}-{args.benchmark}"
    job_name = f"ft-{output_name}"

    # Determine PYTHONPATH based on backend
    if args.policy in LEROBOT_MODELS:
        pythonpath = ":".join([
            str(AGENT_ROOT / "lerobot" / "src"),
            str(AGENT_ROOT / "lerobot"),
        ])
        venv_python = str(AGENT_ROOT / "lerobot" / ".venv" / "bin" / "python3")
    else:
        pythonpath = ":".join([
            str(AGENT_ROOT / "openvla"),
        ])
        venv_python = str(AGENT_ROOT / "openvla" / ".venv" / "bin" / "python3")

    # Build the command string
    if args.policy == "openvla":
        # torchrun command — special handling
        cmd_str = " \\\n    ".join(cmd_parts)
        exec_block = f"PYTHONPATH={pythonpath} {cmd_str}"
    else:
        cmd_str = " \\\n    ".join(cmd_parts)
        exec_block = f"PYTHONPATH={pythonpath} {cmd_str}"

    # Node constraint
    node_line = f"#SBATCH --nodelist={args.node}" if args.node else ""
    exclude_line = f"#SBATCH --exclude={EGL_EXCLUDE_NODES}" if not args.node else ""

    # GPU request
    gpu_line = f"#SBATCH --gres=gpu:{args.num_gpus}"

    # Time limit — rough estimate: ~1hr per 10k steps for single GPU
    hours = max(2, (args.steps // 10000) * args.num_gpus)
    hours = min(hours, 48)  # cap at 48 hours

    log_file = str(LOGS_DIR / f"finetune-{output_name}-%j.log")

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=all
{gpu_line}
#SBATCH --time={hours:02d}:00:00
#SBATCH --output={log_file}
#SBATCH --cpus-per-task={min(args.num_gpus * 8, 32)}
#SBATCH --mem=64G
{node_line}
{exclude_line}

export HF_HOME={os.path.expanduser("~/.cache/huggingface")}
export WANDB_DISABLED=true

echo "=== Finetune: {args.policy} on {args.benchmark} ==="
echo "Node: $(hostname), GPUs: {args.num_gpus}"
echo "Output: {output_dir}"
echo "Steps: {args.steps}, BS: {args.batch_size}, LR: {args.learning_rate}"
echo ""

{exec_block}
TRAIN_EXIT=$?

echo ""
echo "=== Training exit: $TRAIN_EXIT ==="
if [ $TRAIN_EXIT -eq 0 ]; then
    echo "Checkpoints saved to: {output_dir}"
    ls -la {output_dir}/
fi
exit $TRAIN_EXIT
"""

    # Remove empty SBATCH lines
    lines = script.split("\n")
    lines = [l for l in lines if l.strip() != "#SBATCH --nodelist=" and l.strip() != "#SBATCH --exclude="]
    script = "\n".join(lines)

    return script, log_file


def main():
    parser = argparse.ArgumentParser(description="Finetune VLA model on benchmark data")
    parser.add_argument("--policy", required=True, help="Policy name: pi0, pi0.5, smolvla, openvla")
    parser.add_argument("--benchmark", required=True, help="Benchmark/dataset name")
    parser.add_argument("--base_checkpoint", default="", help="Base checkpoint (auto-resolved if empty)")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--learning_rate", default="2e-5")
    parser.add_argument("--output_name", default="")
    parser.add_argument("--node", default="")
    parser.add_argument("--submit", action="store_true", help="Submit as SLURM job")

    args = parser.parse_args()

    # Validate policy
    known_policies = list(LEROBOT_MODELS.keys()) + ["openvla"]
    if args.policy not in known_policies:
        print(f"ERROR: Unknown policy '{args.policy}'. Known: {known_policies}")
        sys.exit(1)

    # Determine output directory (include timestamp to avoid conflicts)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = args.output_name or f"{args.policy}-{args.benchmark}"
    output_dir = str(CHECKPOINT_DIR / f"{output_name}-{ts}")

    print(f"\n=== Finetune Configuration ===")
    print(f"  Policy:     {args.policy}")
    print(f"  Benchmark:  {args.benchmark}")
    print(f"  Backend:    {'LeRobot' if args.policy in LEROBOT_MODELS else 'OpenVLA'}")
    print(f"  Steps:      {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  GPUs:       {args.num_gpus}")
    print(f"  LR:         {args.learning_rate}")
    print(f"  Output:     {output_dir}")

    # Build backend-specific command
    if args.policy in LEROBOT_MODELS:
        cmd_parts, python = build_lerobot_command(args, output_dir)
        base_ckpt = args.base_checkpoint or LEROBOT_MODELS[args.policy]["base_checkpoint"]
    else:
        cmd_parts, python = build_openvla_command(args, output_dir)
        base_ckpt = args.base_checkpoint or OPENVLA_DEFAULTS["base_checkpoint"]

    print(f"  Base ckpt:  {base_ckpt}")

    # Check dataset availability
    if args.policy in LEROBOT_MODELS:
        ds_cfg = LEROBOT_DATASETS.get(args.benchmark)
        if ds_cfg:
            print(f"  Dataset:    {ds_cfg['repo_id']}")
        else:
            print(f"  Dataset:    {args.benchmark} (treated as HF repo_id)")
    else:
        ds_cfg = OPENVLA_DATASETS.get(args.benchmark)
        if ds_cfg:
            print(f"  Dataset:    {ds_cfg['dataset_name']} (RLDS)")
        else:
            print(f"  WARNING: No RLDS dataset mapping for '{args.benchmark}'")

    if args.submit:
        # Generate and submit SLURM job
        script_content, log_file = generate_slurm_script(args, cmd_parts, python, output_dir)

        script_path = str(LOGS_DIR / f"_ft_{output_name}.sh")
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(script_path, "w") as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)

        print(f"\n  SLURM script: {script_path}")
        result = subprocess.run(["sbatch", script_path], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: sbatch failed: {result.stderr}")
            sys.exit(1)

        job_id = result.stdout.strip().split()[-1]
        print(f"  Submitted: {result.stdout.strip()}")
        print(f"  Log: {log_file.replace('%j', job_id)}")
        print(f"\n  Monitor: tail -f {log_file.replace('%j', job_id)}")
        print(f"  Status:  squeue -j {job_id}")
        print(f"\n  After training, evaluate with:")
        print(f"    run_benchmark(policy=\"{args.policy}\", benchmark=\"<target>\", checkpoint=\"{output_dir}/checkpoints/last/pretrained_model\")")

    else:
        # Run locally
        os.makedirs(output_dir, exist_ok=True)
        cmd = " ".join(cmd_parts)
        print(f"\n  Running: {cmd[:200]}...")
        os.execvp(cmd_parts[0], cmd_parts)


if __name__ == "__main__":
    main()
