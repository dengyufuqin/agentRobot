#!/usr/bin/env python3
"""Inspect a local dataset directory and detect its format + metadata.

Supported format detections:
  - LeRobot v2.x (meta/info.json + data/chunk-* parquet files)
  - WebDataset (*.tar shards)
  - Zarr (*.zarr group)
  - RLDS TFRecord (*.tfrecord)
  - HuggingFace parquet dump (data/*.parquet)

Output: JSON with format, episode count, total files, sample modalities.

Exit codes:
  0  detected a known format
  1  bad args
  2  directory not found
  3  format unknown / empty directory
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _first_matching(p: Path, pattern: str, limit: int = 5) -> list[Path]:
    return list(p.rglob(pattern))[:limit]


def detect(root: Path) -> dict:
    report: dict = {"root": str(root), "format": "unknown", "details": {}}

    if (root / "meta" / "info.json").is_file():
        report["format"] = "lerobot"
        try:
            info = json.loads((root / "meta" / "info.json").read_text())
            report["details"] = {
                "codebase_version": info.get("codebase_version"),
                "total_episodes": info.get("total_episodes"),
                "total_frames": info.get("total_frames"),
                "fps": info.get("fps"),
                "features": list(info.get("features", {}).keys()),
            }
        except Exception as e:
            report["details"]["parse_error"] = str(e)
        return report

    tars = _first_matching(root, "*.tar")
    if tars:
        report["format"] = "webdataset"
        report["details"] = {"n_shards_seen": len(tars), "sample": [str(t.relative_to(root)) for t in tars[:3]]}
        return report

    if list(root.glob("*.zarr")) or (root.name.endswith(".zarr") and (root / ".zgroup").is_file()):
        report["format"] = "zarr"
        return report

    tfrs = _first_matching(root, "*.tfrecord")
    if tfrs:
        report["format"] = "rlds_tfrecord"
        report["details"] = {"n_shards_seen": len(tfrs)}
        return report

    parquets = _first_matching(root, "*.parquet")
    if parquets:
        report["format"] = "parquet"
        report["details"] = {"n_parquet": len(parquets), "sample": [str(t.relative_to(root)) for t in parquets[:3]]}
        return report

    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--out", default="", help="Optional JSON output path")
    args = ap.parse_args()

    root = args.dataset_dir.resolve()
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}")
        return 2

    report = detect(root)
    out_s = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(out_s)
    print(out_s)

    if report["format"] == "unknown":
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
