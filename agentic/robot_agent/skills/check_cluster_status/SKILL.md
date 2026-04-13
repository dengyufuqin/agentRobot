---
name: check_cluster_status
description: Check GPU status — SLURM jobs and node availability (HPC), or local nvidia-smi (local mode)
version: 2.0.0
category: cluster
parameters:
  node:
    type: string
    description: "Optional: specific compute node to check GPU usage on (e.g. cn06). If empty, shows overall status."
    required: false
requires:
  bins: [python3]
timeout: 30
command_template: |
  # Auto-detect environment: HPC (SLURM) vs local
  if command -v squeue &>/dev/null; then
    MODE="hpc"
  else
    MODE="local"
  fi

  if [ "$MODE" = "local" ]; then
    echo "=== Local GPU Status ==="
    if command -v nvidia-smi &>/dev/null; then
      nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
      echo ""
      nvidia-smi
    else
      echo "No GPU detected (nvidia-smi not found)"
    fi
  elif [ -z "{node}" ]; then
    echo "=== SLURM Jobs ==="
    squeue -u $USER -o "%.10i %.20j %.8T %.12M %.5D %R %b"
    echo ""
    echo "=== Node Availability ==="
    sinfo -N -o "%N %G %t %e" | head -40
  else
    echo "=== Jobs on {node} ==="
    squeue -u $USER -o "%.10i %.20j %.8T %R" | grep {node} || echo "No jobs on {node}"
    echo ""
    echo "=== GPU Usage on {node} ==="
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {node} "nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader" 2>&1
  fi
---

# Cluster Status Check

Auto-detects environment:
- **HPC mode** (SLURM detected): Shows SLURM jobs, node availability, per-GPU memory on specific nodes
- **Local mode** (no SLURM): Shows local GPU status via nvidia-smi
