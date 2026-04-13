#!/usr/bin/env python3
"""List files matching a glob pattern under a directory.

Used by the agent to discover candidate source files (demo scripts, model
definitions, configs) inside an arbitrary repo before reading them. Skips
common noise directories so the LLM doesn't get drowned in venv files.
"""
import argparse
import fnmatch
import os
import sys
from pathlib import Path

SKIP_DIRS = {
    ".venv", "venv", "env", ".git", "__pycache__", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "node_modules", ".idea", ".vscode",
    "build", "dist", "wheels", "*.egg-info",
    # large data dirs commonly seen in robot-learning repos
    "data", "datasets", "checkpoints", "logs", "wandb", "outputs",
}

MAX_RESULTS = 500

parser = argparse.ArgumentParser()
parser.add_argument("root", help="Directory to walk")
parser.add_argument("--pattern", default="*.py",
                    help="Glob pattern (default: *.py). Use '*' for all files.")
parser.add_argument("--max-depth", type=int, default=6, help="Max recursion depth")
parser.add_argument("--include-hidden", action="store_true",
                    help="Include dotfiles / hidden directories")
args = parser.parse_args()

root = Path(args.root).expanduser().resolve()
if not root.exists():
    print(f"[ERROR] Directory does not exist: {root}", file=sys.stderr)
    sys.exit(2)
if not root.is_dir():
    print(f"[ERROR] Not a directory: {root}", file=sys.stderr)
    sys.exit(2)

pattern = args.pattern
matches = []
truncated = False

def should_skip(name: str) -> bool:
    if not args.include_hidden and name.startswith("."):
        return True
    for skip in SKIP_DIRS:
        if "*" in skip:
            if fnmatch.fnmatch(name, skip):
                return True
        elif name == skip:
            return True
    return False

root_depth = len(root.parts)
for dirpath, dirnames, filenames in os.walk(root):
    dp = Path(dirpath)
    depth = len(dp.parts) - root_depth
    if depth >= args.max_depth:
        dirnames[:] = []  # don't recurse further
        continue

    # prune in-place
    dirnames[:] = [d for d in dirnames if not should_skip(d)]
    dirnames.sort()

    for fname in sorted(filenames):
        if not args.include_hidden and fname.startswith("."):
            continue
        if not fnmatch.fnmatch(fname, pattern):
            continue
        rel = (dp / fname).relative_to(root)
        matches.append(str(rel))
        if len(matches) >= MAX_RESULTS:
            truncated = True
            break
    if truncated:
        break

print(f"[ROOT] {root}")
print(f"[PATTERN] {pattern}")
print(f"[FOUND] {len(matches)} files{' (truncated)' if truncated else ''}")
print("---")
for m in matches:
    print(m)
if truncated:
    print(f"---\n[TRUNCATED at {MAX_RESULTS}] narrow the pattern or pick a subdirectory.")
