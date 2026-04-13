#!/usr/bin/env python3
"""Read a file (or a line range of it) and print it with line numbers.

Used by the agent to inspect arbitrary source files in a repo before
generating a policy_server.py adapter. Output is bounded so the LLM
context window stays manageable.
"""
import argparse
import sys
from pathlib import Path

MAX_LINES = 2000   # hard cap per call
MAX_BYTES = 200_000  # hard cap regardless of lines

parser = argparse.ArgumentParser()
parser.add_argument("file_path")
parser.add_argument("--start", type=int, default=1, help="1-indexed start line")
parser.add_argument("--end", type=int, default=None, help="1-indexed end line (inclusive)")
args = parser.parse_args()

p = Path(args.file_path).expanduser()
if not p.exists():
    print(f"[ERROR] File does not exist: {p}", file=sys.stderr)
    sys.exit(2)
if not p.is_file():
    print(f"[ERROR] Not a regular file: {p}", file=sys.stderr)
    sys.exit(2)

size = p.stat().st_size
if size > MAX_BYTES * 10:
    print(f"[WARN] File is {size} bytes — only first {MAX_BYTES} bytes will be considered")

try:
    with p.open("rb") as f:
        raw = f.read(MAX_BYTES)
except Exception as e:
    print(f"[ERROR] Cannot read {p}: {e}", file=sys.stderr)
    sys.exit(2)

# Decode tolerantly — many ML repos contain non-utf8 in vendored files
try:
    text = raw.decode("utf-8")
except UnicodeDecodeError:
    text = raw.decode("utf-8", errors="replace")

lines = text.splitlines()
total = len(lines)

start = max(1, args.start)
end = args.end if args.end is not None else total
end = min(end, total, start + MAX_LINES - 1)

if start > total:
    print(f"[INFO] start={start} exceeds file length ({total} lines)")
    sys.exit(0)

print(f"[FILE] {p}  ({total} lines, {size} bytes)")
print(f"[RANGE] {start}-{end}")
print("---")
for i in range(start, end + 1):
    print(f"{i:6d}\t{lines[i-1]}")

if end < total:
    print(f"---\n[TRUNCATED] {total - end} more lines. Re-call with --start {end+1} to continue.")
