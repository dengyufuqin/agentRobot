#!/usr/bin/env python3
"""Scan a repo for existing finetune/train scripts BEFORE writing new ones.

Why: agents often default to "write a training loop from scratch" when the
repo already ships a train.py / finetune.py / scripts/train_*.py. That wastes
hours and usually produces a worse version.

Detection:
  1. File-name heuristics: train.py, finetune.py, scripts/train_*.py, tools/train_*.py
  2. Keyword scan inside README for "To train:" / "Finetune:" sections + code blocks
  3. pyproject.toml `[project.scripts]` entries pointing at train/finetune modules

Output JSON:
{
  "found": true,
  "scripts": [
    {"path": "train.py", "kind": "script", "cli_sample": "python train.py --config foo"},
    {"path": "lerobot/scripts/train.py", "kind": "script", "cli_sample": null}
  ],
  "readme_sections": ["Training", "Fine-tuning LeRobot policies"],
  "pyproject_entries": ["lerobot-train = lerobot.scripts.train:main"],
  "recommendation": "Use existing: python lerobot/scripts/train.py --help"
}

Exit codes:
  0 at least one train/finetune script found (don't rewrite)
  3 nothing found (agent may need to write one, or finetuning isn't supported)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


TRAIN_NAMES = re.compile(
    r"(?:^|/)(train|finetune|fine_tune|ft|sft)[_-]?[a-z0-9_]*\.py$",
    re.IGNORECASE,
)
# Headings in README that plausibly contain training/FT instructions
README_SECTION_RE = re.compile(
    r"^#{1,4}\s*(.*(?:train|finetun|fine[- ]tun)[\w\s\-]*)$",
    re.IGNORECASE | re.MULTILINE,
)
# CLI invocations inside code fences
CLI_RE = re.compile(
    r"(?:python3?|uv\s+run)\s+(\S*(?:train|finetune|fine_tune)[\w/.-]*\.py[^\n]*)",
    re.IGNORECASE,
)


def scan_files(repo: Path) -> list[dict]:
    """Find candidate train/finetune .py files."""
    # Depth-limit to avoid walking huge vendored trees
    SKIP_DIRS = {".git", ".venv", "node_modules", "__pycache__", "build", "dist",
                 "outputs", "checkpoints", "wandb", ".probe_io_spec.json"}
    results: list[dict] = []
    for p in repo.rglob("*.py"):
        try:
            rel = p.relative_to(repo)
        except ValueError:
            continue
        parts = set(rel.parts)
        if parts & SKIP_DIRS:
            continue
        if TRAIN_NAMES.search(str(rel).replace("\\", "/")):
            results.append({
                "path": str(rel),
                "kind": "script",
                "cli_sample": None,
            })
    return results


def scan_readme(repo: Path) -> tuple[list[str], list[str]]:
    sections: list[str] = []
    clis: list[str] = []
    for f in list(repo.glob("README*")) + list(repo.glob("docs/*.md"))[:10]:
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        sections.extend(m.group(1).strip() for m in README_SECTION_RE.finditer(text))
        clis.extend(m.group(0).strip() for m in CLI_RE.finditer(text))
    # dedupe preserving order
    def uniq(xs):
        seen = set(); out = []
        for x in xs:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    return uniq(sections)[:10], uniq(clis)[:10]


def scan_pyproject(repo: Path) -> list[str]:
    pp = repo / "pyproject.toml"
    if not pp.is_file():
        return []
    try:
        text = pp.read_text(errors="ignore")
    except Exception:
        return []
    # Very lightweight TOML-ish match — avoid requiring tomllib/toml dep
    entries = re.findall(
        r"^([A-Za-z0-9_-]+)\s*=\s*[\"']([\w\.:]+(?:train|finetune|fine_tune)[\w\.:]*)[\"']",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return [f"{k} = {v}" for k, v in entries]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", required=True, type=Path)
    ap.add_argument("--out", default="", help="Optional output path for JSON report")
    args = ap.parse_args()

    repo = args.repo_path.resolve()
    if not repo.is_dir():
        print(f"ERROR: repo not found: {repo}")
        return 1

    scripts = scan_files(repo)
    sections, clis = scan_readme(repo)
    pp_entries = scan_pyproject(repo)

    # Attach README CLI samples to scripts where filename matches
    for s in scripts:
        base = Path(s["path"]).name
        for cli in clis:
            if base in cli:
                s["cli_sample"] = cli
                break

    found = bool(scripts or pp_entries or clis)
    recommendation = None
    if scripts:
        pref = next((s for s in scripts if "finetune" in s["path"].lower() or "fine_tune" in s["path"].lower()), scripts[0])
        cli = pref["cli_sample"] or f"python3 {pref['path']} --help"
        recommendation = f"Use existing: {cli}"
    elif pp_entries:
        recommendation = f"Use existing console script: {pp_entries[0]}"
    elif clis:
        recommendation = f"Follow README: {clis[0]}"
    else:
        recommendation = "No training script found. Check the repo is a policy library vs. inference-only, then consider writing a minimal trainer or using an external framework (lerobot, hydra)."

    report = {
        "found": found,
        "scripts": scripts,
        "readme_sections": sections,
        "readme_clis": clis,
        "pyproject_entries": pp_entries,
        "recommendation": recommendation,
    }

    out_s = json.dumps(report, indent=2)
    if args.out:
        Path(args.out).write_text(out_s)
    print(out_s)

    return 0 if found else 3


if __name__ == "__main__":
    sys.exit(main())
