#!/usr/bin/env python3
"""Targeted string replacement in a file (atomic, uniqueness-enforced).

The LLM-agent's robust patch primitive. Designed so a weak model can fix a
one-line bug with a ~50-char base64 payload instead of a 6KB full-file rewrite.

Contract (mirrors Claude Code's own `Edit` tool):
  - old_string MUST exist in the file (else: error, no change)
  - old_string MUST be unique unless --replace-all is set (else: error, no change)
  - new_string MUST differ from old_string (else: error)
  - Write is atomic: write to .tmp then rename, so a failed edit can never
    leave the file half-corrupted.

Both --old-b64 and --new-b64 are base64-encoded so the agent can pass
strings containing newlines / quotes / shell metacharacters safely.

Usage:
    edit.py <file_path> --old-b64 <b64> --new-b64 <b64> [--replace-all]
"""
import argparse
import base64
import os
import sys
from pathlib import Path


def _tolerant_b64decode(s: str) -> str:
    """Base64-decode with padding/whitespace fixup. Weak LLMs often drop trailing
    '=' or insert stray whitespace inside base64 payloads — be lenient."""
    s = "".join(s.split())  # strip ALL whitespace, including embedded
    pad = len(s) % 4
    if pad:
        s += "=" * (4 - pad)
    return base64.b64decode(s).decode("utf-8")


def _try_repair(old_s: str, new_s: str, file_text: str):
    """When old_s isn't found in file_text verbatim, try a few cleanup heuristics.
    Returns (repaired_old, repaired_new, description) on success, else None.

    Each repair must produce an old_s that uniquely matches as a substring of
    file_text. We never mutate file_text — only the search/replace strings.
    """
    seen = {old_s}  # don't try a repair that produces the original

    def _attempt(fix_old, fix_new, desc):
        ro = fix_old(old_s)
        if not ro or ro in seen:
            return None
        seen.add(ro)
        if file_text.count(ro) == 1:
            return ro, fix_new(new_s), desc
        return None

    candidates = [
        # Most common LLM bug: literal newline inserted between adjacent chars
        # (e.g. "GPTConfig" → "GPT\nConfig" during base64 emission)
        (lambda s: s.replace("\n", ""),
         lambda s: s.replace("\n", ""),
         "stripped embedded newlines"),
        # LLM sometimes embeds the literal two-char sequence "\n" instead of \n
        (lambda s: s.replace("\\n", ""),
         lambda s: s.replace("\\n", ""),
         "stripped literal '\\n' sequences"),
        # Strip all whitespace runs to single spaces (catches both \n and \t glitches)
        (lambda s: s.replace("\n", "").replace("\t", "").replace("\r", ""),
         lambda s: s.replace("\n", "").replace("\t", "").replace("\r", ""),
         "stripped \\n\\t\\r"),
        # LLM stripped leading/trailing whitespace from a multi-line snippet
        (lambda s: s.strip(),
         lambda s: s.strip(),
         "stripped leading/trailing whitespace"),
    ]
    for fix_old, fix_new, desc in candidates:
        result = _attempt(fix_old, fix_new, desc)
        if result is not None:
            return result
    return None


parser = argparse.ArgumentParser()
parser.add_argument("file_path")
parser.add_argument("--old-b64", required=True, help="Base64-encoded string to find")
parser.add_argument("--new-b64", required=True, help="Base64-encoded replacement string")
parser.add_argument("--replace-all", action="store_true",
                    help="Replace all occurrences (default: must be unique)")
args = parser.parse_args()

p = Path(args.file_path).expanduser()
if not p.exists():
    print(f"[ERROR] file does not exist: {p}", file=sys.stderr)
    sys.exit(2)
if not p.is_file():
    print(f"[ERROR] not a regular file: {p}", file=sys.stderr)
    sys.exit(2)

try:
    old_string = _tolerant_b64decode(args.old_b64)
except Exception as e:
    print(f"[ERROR] could not decode --old-b64: {e}", file=sys.stderr)
    sys.exit(2)
try:
    new_string = _tolerant_b64decode(args.new_b64)
except Exception as e:
    print(f"[ERROR] could not decode --new-b64: {e}", file=sys.stderr)
    sys.exit(2)

if old_string == new_string:
    print("[ERROR] old_string and new_string are identical — nothing to do", file=sys.stderr)
    sys.exit(2)

if not old_string:
    print("[ERROR] old_string is empty (would replace everything)", file=sys.stderr)
    sys.exit(2)

# Read original
try:
    original = p.read_text(encoding="utf-8")
except UnicodeDecodeError:
    print(f"[ERROR] file is not valid UTF-8: {p}", file=sys.stderr)
    sys.exit(2)

# Count occurrences and validate uniqueness
count = original.count(old_string)
repair_note = None
if count == 0:
    # Try a few cleanup heuristics — the most common LLM bug is a stray newline
    # inserted between adjacent characters during base64 emission. We never
    # mutate the file; we only try variants of old_string that uniquely match.
    repair = _try_repair(old_string, new_string, original)
    if repair is not None:
        old_string, new_string, repair_desc = repair
        repair_note = f"[REPAIR] auto-repaired old_b64: {repair_desc} (now uniquely matches)"
        print(repair_note, file=sys.stderr)
        count = original.count(old_string)
    else:
        print(f"[ERROR] old_string NOT FOUND in {p}", file=sys.stderr)
        print(f"[HINT] Check whitespace/indentation. The string must match EXACTLY,", file=sys.stderr)
        print(f"       including leading spaces. Re-read the file with read_file to confirm.", file=sys.stderr)
        print(f"[HINT] Tried auto-repair (strip \\n, strip whitespace) — none produced a unique match.", file=sys.stderr)
        sys.exit(1)

if count > 1 and not args.replace_all:
    print(f"[ERROR] old_string matches {count} places — not unique", file=sys.stderr)
    print(f"[HINT] Either:", file=sys.stderr)
    print(f"       (a) include more surrounding context in old_string to disambiguate, or", file=sys.stderr)
    print(f"       (b) re-call with --replace-all if you really want all {count} replaced.", file=sys.stderr)
    sys.exit(1)

# Perform replacement
if args.replace_all:
    new_content = original.replace(old_string, new_string)
    n_replaced = count
else:
    new_content = original.replace(old_string, new_string, 1)
    n_replaced = 1

# Atomic write: tmp file in same dir, then rename
tmp_path = p.with_suffix(p.suffix + ".edit.tmp")
try:
    tmp_path.write_text(new_content, encoding="utf-8")
    os.replace(str(tmp_path), str(p))
except Exception as e:
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass
    print(f"[ERROR] write failed: {e}", file=sys.stderr)
    sys.exit(2)

# Report what changed: find the line of the first replacement and show context
lines_old = original.splitlines()
lines_new = new_content.splitlines()

# Find the first line that differs (1-indexed)
first_diff = None
for i in range(min(len(lines_old), len(lines_new))):
    if lines_old[i] != lines_new[i]:
        first_diff = i + 1
        break
if first_diff is None and len(lines_old) != len(lines_new):
    first_diff = min(len(lines_old), len(lines_new)) + 1

print(f"[OK] {p}")
if repair_note:
    print(repair_note)
print(f"[REPLACED] {n_replaced} occurrence{'s' if n_replaced != 1 else ''}")
if first_diff is not None:
    lo = max(1, first_diff - 2)
    hi = min(len(lines_new), first_diff + 4)
    print(f"[CONTEXT] lines {lo}-{hi} after edit:")
    for i in range(lo, hi + 1):
        marker = ">" if i == first_diff else " "
        print(f"  {marker} {i:6d}\t{lines_new[i-1]}")
print(f"[SIZE] {len(original)} → {len(new_content)} bytes")
