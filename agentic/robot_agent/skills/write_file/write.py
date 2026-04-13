#!/usr/bin/env python3
"""Write content to a file from base64-encoded input."""
import argparse
import base64
import sys

parser = argparse.ArgumentParser()
parser.add_argument("file_path")
parser.add_argument("--b64", help="Base64-encoded content")
parser.add_argument("--executable", action="store_true")
args = parser.parse_args()

content = base64.b64decode(args.b64).decode("utf-8")

with open(args.file_path, "w") as f:
    f.write(content)

if args.executable:
    import os
    os.chmod(args.file_path, 0o755)

lines = content.count("\n") + 1
print(f"Written {lines} lines to {args.file_path}")
print(f"Size: {len(content)} bytes")
