#!/usr/bin/env python3
"""Download a HuggingFace model with *checkpoint-variant* awareness.

Motivation:
  A common failure we've seen: user says "I want pi0 on libero_spatial", the
  agent downloads `lerobot/pi0_libero` (the BASE checkpoint) instead of
  `lerobot/pi0_libero_finetuned` (the task-finetuned one), and the benchmark
  scores 0%. The fix is to make the choice explicit up front.

What this does:
  1. If --repo-id is an exact HF ID and exists → download.
  2. If --repo-id is ambiguous ("pi0 libero") → list sibling repos that
     match the pattern, print them, and exit 10 so the caller can re-invoke
     with a concrete ID (interactive mode uses the `variants` output).
  3. With --prefer-finetuned true, auto-pick the repo whose name contains
     "finetuned" when multiple candidates exist.

Exit codes:
  0  download complete
  1  bad args / HF auth missing
  2  repo does not exist
  10 ambiguous: multiple variants found — caller should disambiguate
  11 download failed mid-stream
"""
from __future__ import annotations

import argparse
import json
import re
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-id", required=True,
                    help="HF repo ID (e.g. 'lerobot/pi0_libero_finetuned') OR a pattern ('pi0 libero')")
    ap.add_argument("--local-dir", default="",
                    help="Target dir; if empty uses HF cache")
    ap.add_argument("--prefer-finetuned", default="true",
                    help="If multiple variants match, prefer the one containing 'finetuned'")
    ap.add_argument("--allow-patterns", default="",
                    help="Comma-separated glob patterns to restrict download (e.g. '*.safetensors,config.json')")
    ap.add_argument("--list-only", default="false",
                    help="'true' to just list matching variants and exit")
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. pip install huggingface_hub")
        return 1

    api = HfApi()
    prefer_ft = args.prefer_finetuned.lower() == "true"

    # Is it an exact ID? (contains a slash, no spaces)
    looks_exact = "/" in args.repo_id and " " not in args.repo_id

    candidates: list[str] = []
    if looks_exact:
        try:
            api.model_info(args.repo_id)
            candidates = [args.repo_id]
        except Exception as e:
            # Maybe it's a near-miss: search siblings
            print(f"[download_model] exact repo not found: {args.repo_id} ({e})")
            # Try pattern search below
            looks_exact = False

    if not looks_exact:
        # Build a case-insensitive AND-match of all whitespace-separated tokens
        tokens = [t for t in re.split(r"[\s/]+", args.repo_id) if t]
        print(f"[download_model] searching HF for tokens: {tokens}")
        matched = []
        for m in api.list_models(search=" ".join(tokens), limit=50):
            if all(re.search(re.escape(t), m.id, re.IGNORECASE) for t in tokens):
                matched.append(m.id)
        candidates = matched[:20]

    if not candidates:
        print(f"[download_model] no repos match '{args.repo_id}'")
        return 2

    if args.list_only.lower() == "true":
        print(json.dumps({"variants": candidates}, indent=2))
        return 0

    if len(candidates) > 1:
        # Trained-on-target indicators that show up in HF repo names.
        # finetune / ft / sft (supervised finetuning) / lora / dpo / rl(hf) / grpo / pretrain
        FT_PAT = re.compile(r"finetune|finetuned|\bft\b|\bsft\b|lora|\bdpo\b|grpo|rlhf|pretrain", re.IGNORECASE)
        ft = [c for c in candidates if FT_PAT.search(c)]
        base = [c for c in candidates if c not in ft]
        if prefer_ft and ft:
            # If exactly one finetuned match, use it; else ambiguous
            if len(ft) == 1:
                pick = ft[0]
                print(f"[download_model] prefer-finetuned picked: {pick}")
                print(f"[download_model] (other candidates: {ft[1:] + base})")
            else:
                # Order matters: finetuned first so a numeric pick prefers them.
                ordered = ft + base
                print(json.dumps({"ambiguous": True, "variants": ordered,
                                   "finetuned": ft, "base": base}, indent=2))
                return 10
        else:
            ordered = ft + base
            print(json.dumps({"ambiguous": True, "variants": ordered,
                               "finetuned": ft, "base": base}, indent=2))
            return 10
    else:
        pick = candidates[0]

    if 'pick' not in locals():
        pick = candidates[0]

    allow = None
    if args.allow_patterns:
        allow = [p.strip() for p in args.allow_patterns.split(",") if p.strip()]

    local = args.local_dir or None
    print(f"[download_model] downloading {pick}" + (f" → {local}" if local else " (HF cache)"))
    try:
        path = snapshot_download(
            repo_id=pick,
            local_dir=local,
            allow_patterns=allow,
        )
        print(f"[download_model] DONE: {path}")

        # Auto-patch missing base-model config files. Many community-finetuned
        # checkpoints ship only weights + the modified config.json — they expect
        # the auto_map paths (e.g. configuration_prismatic.py) to be resolvable
        # but never copy them in. Detect this and grab from the canonical base.
        import os, shutil
        from huggingface_hub import hf_hub_download as _hf_dl
        BASE_PATCHES = [
            (lambda p: "openvla" in pick.lower(),
             "openvla/openvla-7b",
             ["configuration_prismatic.py", "modeling_prismatic.py", "processing_prismatic.py"]),
        ]
        for cond, base_repo, files in BASE_PATCHES:
            if not cond(pick):
                continue
            patched = []
            for fname in files:
                if not os.path.isfile(os.path.join(path, fname)):
                    try:
                        src = _hf_dl(base_repo, fname)
                        shutil.copy(src, os.path.join(path, fname))
                        patched.append(fname)
                    except Exception:
                        pass
            if patched:
                print(f"[download_model] auto-patched from {base_repo}: {patched}")

        print(json.dumps({"repo_id": pick, "local_path": str(path)}))
        return 0
    except Exception as e:
        print(f"[download_model] download failed: {e}")
        return 11


if __name__ == "__main__":
    sys.exit(main())
