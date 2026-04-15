#!/usr/bin/env python3
"""Emit a minimal, working `make_dataloader()` factory file for a given format.

Intentionally restricted scope: we produce a *starter template* the LLM can
further refine. The goal is to catch the 80% case (LeRobot v2) fast and give
the LLM clear extension points for the rest.

Exit codes:
  0  template written
  1  bad args
  2  format unsupported
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


LEROBOT_TEMPLATE = '''"""Auto-generated dataloader factory for a LeRobot v2 dataset.

Customize as needed — this is a starter. The key contract is:
    make_dataloader(**kwargs) -> torch.utils.data.DataLoader
yielding dict batches with image/state/action keys.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lerobot >= 0.5
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # legacy


def make_dataloader(
    repo_id: str = {repo_id!r},
    root: str = {root!r},
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = True,
    delta_timestamps: dict | None = None,
) -> DataLoader:
    ds = LeRobotDataset(
        repo_id=repo_id,
        root=root if root else None,
        delta_timestamps=delta_timestamps,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=None,
    )


if __name__ == "__main__":
    dl = make_dataloader()
    b = next(iter(dl))
    if isinstance(b, dict):
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                print(f"{{k}}: shape={{tuple(v.shape)}} dtype={{v.dtype}}")
            else:
                print(f"{{k}}: {{type(v).__name__}}")
    else:
        print(f"batch type: {{type(b).__name__}}")
'''


PARQUET_TEMPLATE = '''"""Auto-generated dataloader factory for a parquet-format dataset."""
from __future__ import annotations

import glob
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset


class ParquetDataset(Dataset):
    def __init__(self, root: str):
        self.files = sorted(glob.glob(f"{{root}}/**/*.parquet", recursive=True))
        if not self.files:
            raise FileNotFoundError(f"no parquet files under {{root}}")
        self._tables = [pq.read_table(f) for f in self.files]
        self._row_to_file = []
        for i, t in enumerate(self._tables):
            self._row_to_file.extend([(i, j) for j in range(t.num_rows)])

    def __len__(self):
        return len(self._row_to_file)

    def __getitem__(self, idx):
        fi, ri = self._row_to_file[idx]
        row = self._tables[fi].slice(ri, 1).to_pydict()
        return {{k: v[0] for k, v in row.items()}}


def make_dataloader(root: str = {root!r}, batch_size: int = 2, num_workers: int = 0) -> DataLoader:
    ds = ParquetDataset(root)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
'''


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", required=True, choices=["lerobot", "parquet"],
                    help="Dataset format (from validate_dataset)")
    ap.add_argument("--repo-id", default="", help="HF dataset repo ID (lerobot format)")
    ap.add_argument("--root", default="", help="Local dataset root path")
    ap.add_argument("--out", required=True, help="Where to write the factory file")
    args = ap.parse_args()

    if args.format == "lerobot":
        code = LEROBOT_TEMPLATE.format(repo_id=args.repo_id, root=args.root)
    elif args.format == "parquet":
        code = PARQUET_TEMPLATE.format(root=args.root)
    else:
        print(f"ERROR: unsupported format: {args.format}")
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(code)
    print(f"[generate_dataloader] wrote {out_path}")
    print(f"[generate_dataloader] next: validate_dataloader(factory_module={out_path}, factory_func=make_dataloader)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
