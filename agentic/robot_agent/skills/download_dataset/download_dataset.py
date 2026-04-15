#!/usr/bin/env python3
"""Download a HuggingFace *dataset* to a chosen local dir.

Differences from download_model:
  - Uses HfApi.dataset_info / snapshot_download(repo_type='dataset').
  - Accepts --splits to limit to certain splits via allow_patterns.
  - Reports on-disk size after download.

Exit codes:
  0  downloaded
  1  bad args / auth
  2  repo not found
  11 download failed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _dir_size(p: Path) -> int:
    total = 0
    for f in p.rglob("*"):
        if f.is_file():
            try:
                total += f.stat().st_size
            except OSError:
                pass
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True, help="HF dataset ID (e.g. 'lerobot/libero_spatial')")
    ap.add_argument("--local-dir", default="", help="Target dir; empty = HF cache")
    ap.add_argument("--allow-patterns", default="",
                    help="Comma-separated globs (e.g. 'data/train/*,meta/*')")
    ap.add_argument("--max-files", type=int, default=0,
                    help="Soft cap on download — 0 = no cap (applied via check after listing)")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed")
        return 1

    api = HfApi()
    try:
        info = api.dataset_info(args.repo_id)
    except Exception as e:
        print(f"ERROR: dataset not found: {args.repo_id} ({e})")
        return 2

    print(f"[download_dataset] repo: {args.repo_id}")
    print(f"[download_dataset] last modified: {info.last_modified}")
    siblings = info.siblings if info.siblings is not None else []
    print(f"[download_dataset] files: {len(siblings)}")

    allow = None
    if args.allow_patterns:
        allow = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]

    local = args.local_dir or None
    if local and Path(local).exists() and any(Path(local).iterdir()):
        existing = _dir_size(Path(local))
        print(f"[download_dataset] target dir exists ({existing / 1e9:.2f} GB) — will reuse/resume")

    try:
        path = snapshot_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            local_dir=local,
            allow_patterns=allow,
        )
        size_gb = _dir_size(Path(path)) / 1e9
        print(f"[download_dataset] DONE: {path} ({size_gb:.2f} GB)")
        print(json.dumps({"repo_id": args.repo_id, "local_path": str(path), "size_gb": round(size_gb, 2)}))
        return 0
    except Exception as e:
        print(f"[download_dataset] failed: {e}")
        return 11


if __name__ == "__main__":
    sys.exit(main())
