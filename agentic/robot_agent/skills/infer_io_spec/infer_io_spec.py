#!/usr/bin/env python3
"""Resolve a repo's obs/action spec from THREE sources, in priority order:

  1. README/docs: scan for shape/dim patterns (`action_dim=7`, `(3, 224, 224)`, ...)
  2. probe_run capture: .probe_io_spec.json if it exists
  3. User fallback: a JSON file the user/agent can provide

Why three: a repo's README is often wrong or missing. probe_run gives ground truth
from the code. But probe_run can also pick up the vision encoder (submodule), not
the policy head. Merging sources lets each fix the others' blind spots.

Output JSON:
{
  "action_dim": 7,
  "action_shape": [1, 50, 7],
  "image_shape": [3, 224, 224],
  "state_dim": 8,
  "sources_used": ["probe", "readme"],
  "conflicts": ["readme says 6 but probe says 7"],
  "confidence": "high" | "medium" | "low"
}
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


def scan_readme(repo_path: Path) -> dict:
    """Heuristic extraction from README/docs."""
    hits: dict = {"source": "readme", "image_shape": None, "action_dim": None, "state_dim": None}
    candidates = list(repo_path.glob("README*")) + list(repo_path.glob("docs/*.md"))[:10]

    action_patterns = [
        re.compile(r"action[_\s]*dim[:=\s]+(\d+)", re.IGNORECASE),
        re.compile(r"(\d+)-?dim(?:ensional)?\s+action", re.IGNORECASE),
        re.compile(r"action.*shape[:\s]*\[?\(?([\d,\s]+)\)?\]?", re.IGNORECASE),
    ]
    state_patterns = [
        re.compile(r"state[_\s]*dim[:=\s]+(\d+)", re.IGNORECASE),
        re.compile(r"proprio[_\s]*dim[:=\s]+(\d+)", re.IGNORECASE),
    ]
    image_patterns = [
        re.compile(r"(\d+)\s*x\s*(\d+)\s*(?:RGB|image|pixels?)", re.IGNORECASE),
        re.compile(r"image.*resolution[:\s]*(\d+)", re.IGNORECASE),
    ]

    for f in candidates:
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for p in action_patterns:
            m = p.search(text)
            if m and not hits["action_dim"]:
                try:
                    hits["action_dim"] = int(m.group(1).split(",")[-1].strip())
                except Exception:
                    pass
        for p in state_patterns:
            m = p.search(text)
            if m and not hits["state_dim"]:
                try:
                    hits["state_dim"] = int(m.group(1))
                except Exception:
                    pass
        for p in image_patterns:
            m = p.search(text)
            if m and not hits["image_shape"]:
                try:
                    dim = int(m.group(1))
                    hits["image_shape"] = [3, dim, dim]
                except Exception:
                    pass

    return hits


def from_probe(spec_file: Path) -> dict:
    if not spec_file.is_file():
        return {"source": "probe", "error": "spec file missing"}
    try:
        calls = json.loads(spec_file.read_text())
    except Exception as e:
        return {"source": "probe", "error": f"parse failed: {e}"}
    if not isinstance(calls, list) or not calls:
        return {"source": "probe", "error": "empty or non-list"}

    out: dict = {"source": "probe", "image_shape": None, "action_dim": None, "state_dim": None,
                 "calls": []}

    # Prefer calls NOT from encoder-like submodules
    ENCODERS = ("Siglip", "VisionTower", "ViT", "CLIP", "Tokenizer", "Embedding", "Backbone")

    def is_encoder(name: str) -> bool:
        return any(e in name for e in ENCODERS)

    policy_calls = [c for c in calls if not is_encoder(c.get("module_class", ""))]
    preferred = policy_calls or calls

    for c in preferred:
        out["calls"].append({
            "module": c.get("module_class"),
            "source_tag": c.get("source"),
            "encoder_like": is_encoder(c.get("module_class", "")),
        })
        # Walk args to find image (4D, W>=64) and state (2D small last-dim)
        def walk(x, _out):
            if isinstance(x, dict):
                if "shape" in x:
                    s = x["shape"]
                    if len(s) == 4 and s[-1] >= 64 and not _out.get("image_shape"):
                        _out["image_shape"] = s[-3:]
                    elif len(s) == 2 and s[-1] <= 32 and not _out.get("state_dim"):
                        _out["state_dim"] = s[-1]
                    return
                for v in x.values():
                    walk(v, _out)
            elif isinstance(x, list):
                for v in x:
                    walk(v, _out)
        for a in c.get("input_args", []):
            walk(a, out)
        for v in c.get("input_kwargs", {}).values():
            walk(v, out)
        # Output action dim — last dim of the policy output
        o = c.get("output")
        if isinstance(o, dict) and "shape" in o:
            s = o["shape"]
            if not out["action_dim"] and 1 <= s[-1] <= 64:
                out["action_dim"] = s[-1]

    return out


def merge(sources: list[dict]) -> dict:
    """Merge with priority: probe > user > readme. Report conflicts."""
    # Sort priority
    PRIO = {"user": 0, "probe": 1, "readme": 2}
    sources = sorted(sources, key=lambda s: PRIO.get(s.get("source") or "", 99))

    final: dict = {"action_dim": None, "image_shape": None, "state_dim": None,
                   "sources_used": [], "conflicts": [], "per_source": sources}

    for key in ("action_dim", "image_shape", "state_dim"):
        for s in sources:
            v = s.get(key)
            if v is None:
                continue
            if final[key] is None:
                final[key] = v
                final["sources_used"].append(f"{key}←{s['source']}")
            elif final[key] != v:
                final["conflicts"].append(
                    f"{key}: {final['sources_used'][-1]} says {final[key]}, {s['source']} says {v}"
                )

    # Confidence heuristic
    got = sum(1 for k in ("action_dim", "image_shape", "state_dim") if final[k] is not None)
    if got == 3 and not final["conflicts"]:
        final["confidence"] = "high"
    elif got >= 2:
        final["confidence"] = "medium"
    else:
        final["confidence"] = "low"
    return final


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", required=True, type=Path)
    ap.add_argument("--probe-spec-file", default="",
                    help="Path to .probe_io_spec.json (default: {repo_path}/.probe_io_spec.json)")
    ap.add_argument("--user-fallback", default="",
                    help="Optional JSON file with user-provided {action_dim, image_shape, state_dim}")
    ap.add_argument("--out", default="", help="Optional output path for merged spec")
    args = ap.parse_args()

    repo = args.repo_path.resolve()
    if not repo.is_dir():
        print(f"ERROR: repo not found: {repo}")
        return 1

    sources: list[dict] = []

    readme_spec = scan_readme(repo)
    if any(readme_spec.get(k) for k in ("action_dim", "image_shape", "state_dim")):
        sources.append(readme_spec)
    else:
        sources.append({"source": "readme", "error": "no patterns matched"})

    probe_file = Path(args.probe_spec_file) if args.probe_spec_file else repo / ".probe_io_spec.json"
    sources.append(from_probe(probe_file))

    if args.user_fallback:
        try:
            user = json.loads(Path(args.user_fallback).read_text())
            user["source"] = "user"
            sources.append(user)
        except Exception as e:
            sources.append({"source": "user", "error": f"could not read: {e}"})

    # Filter out error-only entries before merging (but keep for reporting)
    usable = [s for s in sources if not s.get("error")]
    merged = merge(usable) if usable else {"error": "no source produced data", "per_source": sources}

    out_s = json.dumps(merged, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(out_s)
    print(out_s)

    if merged.get("confidence") == "low" or merged.get("error"):
        print("\n[infer_io_spec] LOW confidence — user should provide fallback", file=sys.stderr)
        return 3
    if merged.get("conflicts"):
        print(f"\n[infer_io_spec] WARN — conflicts: {merged['conflicts']}", file=sys.stderr)
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
