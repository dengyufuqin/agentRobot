#!/usr/bin/env python3
"""
Robot Ops Agent — OpenClaw-inspired agentic orchestration for robot learning.

Reads SKILL.md files, builds Claude tool definitions, and runs a ReAct loop
to execute user commands via natural language.

Usage:
    python robot_agent/agent.py "check what GPUs are available on cn06"
    python robot_agent/agent.py   # interactive mode
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# SKILL.md loader
# ---------------------------------------------------------------------------

def parse_skill_md(path: Path) -> dict:
    """Parse a SKILL.md file into a skill definition."""
    text = path.read_text()

    # Split YAML frontmatter from body
    match = re.match(r"^---\n(.*?)\n---\n?(.*)", text, re.DOTALL)
    if not match:
        raise ValueError(f"No YAML frontmatter in {path}")

    frontmatter = yaml.safe_load(match.group(1))
    body = match.group(2).strip()

    # Resolve script references relative to skill directory
    skill_dir = path.parent

    return {
        "name": frontmatter["name"],
        "description": frontmatter["description"],
        "version": frontmatter.get("version", "0.0.0"),
        "category": frontmatter.get("category", "general"),
        "parameters": frontmatter.get("parameters", {}),
        "command_template": frontmatter.get("command_template", ""),
        "requires": frontmatter.get("requires", {}),
        "timeout": frontmatter.get("timeout", 120),
        "notes": body,
        "source": str(path),
        "skill_dir": str(skill_dir),
    }


def load_skills(skills_dir: Path) -> list[dict]:
    """Load skills from directory — supports both flat .md files and subdirectory/SKILL.md."""
    skills = []
    seen = set()

    # Priority 1: subdirectory-based skills (OpenClaw style)
    for subdir in sorted(skills_dir.iterdir()):
        if subdir.is_dir():
            skill_md = subdir / "SKILL.md"
            if skill_md.exists():
                try:
                    skill = parse_skill_md(skill_md)
                    skills.append(skill)
                    seen.add(skill["name"])
                except Exception as e:
                    print(f"  [WARN] Failed to load {subdir.name}/SKILL.md: {e}")

    # Priority 2: flat .md files (backward compat, skip if already loaded)
    for path in sorted(skills_dir.glob("*.md")):
        try:
            skill = parse_skill_md(path)
            if skill["name"] not in seen:
                skills.append(skill)
                seen.add(skill["name"])
        except Exception as e:
            print(f"  [WARN] Failed to load {path.name}: {e}")

    return skills


def discover_policy_servers(base_dir: Path) -> list[dict]:
    """Auto-discover repos with policy_server.yaml and generate deploy skills."""
    discovered = []
    base_dir = base_dir.resolve()  # Ensure absolute paths
    for yaml_path in sorted(base_dir.glob("*/policy_server.yaml")):
        try:
            meta = yaml.safe_load(yaml_path.read_text())
            name = meta["name"]
            ps = meta["policy_server"]
            res = ps.get("resources", {})
            setup = ps.get("setup", {})
            repo_path = str(yaml_path.parent.resolve())

            # Build env setup commands
            env_lines = []
            if setup.get("env_activate"):
                env_lines.append(setup["env_activate"])
            base = str(base_dir)
            if setup.get("pythonpath"):
                pp = ":".join(f"{base}/{p}" for p in setup["pythonpath"])
                env_lines.append(f"export PYTHONPATH={pp}")
            for k, v in setup.get("env_vars", {}).items():
                env_lines.append(f"export {k}={v}")
            env_block = "\n    ".join(env_lines)

            # Build server launch args
            arg_parts = []
            for arg in ps.get("arguments", []):
                if arg["name"] == "port":
                    arg_parts.append(f"{arg['flag']} {{port}}")
                else:
                    arg_parts.append(f"{arg['flag']} {{{arg['name']}}}")
            args_str = " \\\n      ".join(arg_parts)

            # Build parameters dict
            parameters = {
                "node": {"type": "string", "description": "Compute node (e.g. cn06)", "required": True},
                "port": {"type": "integer", "description": "WebSocket port", "default": 18800},
                "gpu_id": {"type": "integer", "description": "GPU device ID (0-7)", "default": 7},
            }
            for arg in ps.get("arguments", []):
                if arg["name"] not in ("port",):
                    parameters[arg["name"]] = {
                        "type": arg.get("type", "string"),
                        "description": arg.get("flag", arg["name"]),
                        "default": arg.get("default", ""),
                    }

            cmd = f"""REPO={repo_path}
LOG={base}/logs/{name}-{{node}}-{{port}}.log

if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 {{node}} "ss -tlnp | grep :{{port}}" 2>/dev/null | grep -q :{{port}}; then
  echo "ERROR: Port {{port}} already in use on {{node}}"
  exit 1
fi

ssh -o StrictHostKeyChecking=no {{node}} "
  cd $REPO
  {env_block}
  export CUDA_VISIBLE_DEVICES={{gpu_id}}
  nohup python3 {ps['entry_point']} \\
    {args_str} \\
    > $LOG 2>&1 &
  echo \\"PID=\\$!\\"
"
echo "Server starting on {{node}}:{{port}} (GPU {{gpu_id}}). Log: $LOG"
echo "Wait ~{res.get('startup_seconds', 90)}s for model to load."
"""

            skill = {
                "name": f"deploy_{name}",
                "description": f"Deploy {name} policy server — {meta.get('description', '')}",
                "version": "auto",
                "category": "deploy",
                "parameters": parameters,
                "command_template": cmd,
                "requires": {"bins": ["ssh"]},
                "timeout": 120,
                "notes": f"Auto-discovered from {yaml_path}. GPU: ~{res.get('gpu_memory_gb', '?')}GB, "
                         f"load: ~{res.get('startup_seconds', '?')}s, action_dim: {res.get('action_dim', '?')}",
                "source": str(yaml_path),
                "skill_dir": repo_path,
            }
            discovered.append(skill)
        except Exception as e:
            print(f"  [WARN] Failed to parse {yaml_path}: {e}")
    return discovered


def skill_to_claude_tool(skill: dict) -> dict:
    """Convert a skill definition to a Claude API tool definition."""
    properties = {}
    required = []

    for param_name, param_def in skill["parameters"].items():
        prop = {
            "type": param_def.get("type", "string"),
            "description": param_def.get("description", ""),
        }
        if "default" in param_def:
            prop["description"] += f" (default: {param_def['default']})"
        properties[param_name] = prop
        if param_def.get("required", False):
            required.append(param_name)

    return {
        "name": skill["name"],
        "description": skill["description"] + "\n\n" + skill["notes"],
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


# ---------------------------------------------------------------------------
# Skill executor
# ---------------------------------------------------------------------------

def execute_skill(skill: dict, args: dict) -> str:
    """Execute a skill's command template with the given arguments."""
    cmd = skill["command_template"]
    timeout = skill.get("timeout", 120)

    # Fill in defaults for missing optional params
    for param_name, param_def in skill["parameters"].items():
        if param_name not in args and "default" in param_def:
            args[param_name] = param_def["default"]

    # Substitute parameters — only replace known parameter names
    known_params = set(skill["parameters"].keys())
    for key, value in args.items():
        cmd = cmd.replace(f"{{{key}}}", str(value))

    # Only clear unfilled KNOWN optional params (not arbitrary {word} patterns)
    for param_name in known_params:
        if param_name not in args:
            cmd = cmd.replace(f"{{{param_name}}}", "")

    # Use skill_dir as working directory if available
    cwd = skill.get("skill_dir", None)

    print(f"  [EXEC] Running skill: {skill['name']} (timeout={timeout}s)")
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[STDERR]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[EXIT CODE: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[ERROR] Command timed out after {timeout} seconds"
    except Exception as e:
        return f"[ERROR] {e}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

def _load_soul(soul_path: Path = None) -> str:
    """Load SOUL.md — the agent's identity and rules (OpenClaw pattern)."""
    if soul_path is None:
        soul_path = Path(__file__).parent / "SOUL.md"
    if soul_path.exists():
        return soul_path.read_text().strip()
    return "You are Robot Ops Agent — an AI assistant for managing robot learning systems."


SYSTEM_PROMPT = _load_soul()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _detect_provider():
    """Auto-detect which LLM provider is available."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY"):
        return "openai"
    return None


def _call_anthropic(messages, tools, model):
    """Call Claude API."""
    import anthropic
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model, max_tokens=4096, system=SYSTEM_PROMPT,
        tools=tools, messages=messages,
    )
    # Normalize to common format
    text_parts = []
    tool_calls = []
    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({"id": block.id, "name": block.name, "arguments": block.input})
    return text_parts, tool_calls, response.content, response.stop_reason == "end_turn", []


def _try_repair_json(raw: str):
    """Attempt to repair malformed JSON from LLM output."""
    import re as _re
    # Strategy 1: truncated JSON — try adding closing braces
    for suffix in ["}", "}}", '"}'  , '"}', '"}}'  ]:
        try:
            return json.loads(raw + suffix)
        except json.JSONDecodeError:
            pass
    # Strategy 2: extract first valid JSON object
    m = _re.search(r'\{[^{}]*\}', raw)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    # Strategy 3: the args are just a simple string that should be wrapped
    try:
        return json.loads('{"input": ' + json.dumps(raw) + '}')
    except Exception:
        pass
    return None


def _call_openai(messages, tools, model):
    """Call OpenAI API (compatible with any OpenAI-format API, including Dashscope/Qwen)."""
    from openai import OpenAI

    dashscope_key = os.environ.get("DASHSCOPE_API_KEY")
    if dashscope_key and not os.environ.get("OPENAI_API_KEY"):
        client = OpenAI(
            api_key=dashscope_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
    else:
        client = OpenAI()

    # Convert tools to OpenAI format
    oai_tools = []
    for t in tools:
        oai_tools.append({
            "type": "function",
            "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]},
        })

    # Convert messages: Anthropic format → OpenAI format
    oai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in messages:
        if msg["role"] == "user":
            if isinstance(msg["content"], str):
                oai_messages.append({"role": "user", "content": msg["content"]})
            elif isinstance(msg["content"], list):
                # Tool results
                for tr in msg["content"]:
                    oai_messages.append({
                        "role": "tool",
                        "tool_call_id": tr["tool_use_id"],
                        "content": tr["content"],
                    })
        elif msg["role"] == "assistant":
            raw = msg["content"]
            if isinstance(raw, str):
                oai_messages.append({"role": "assistant", "content": raw})
            elif hasattr(raw, "role"):
                # OpenAI ChatCompletionMessage object — pass through directly
                m = {"role": "assistant"}
                if raw.content:
                    m["content"] = raw.content
                if raw.tool_calls:
                    m["tool_calls"] = [
                        {"id": t.id, "type": "function",
                         "function": {"name": t.function.name, "arguments": t.function.arguments}}
                        for t in raw.tool_calls
                    ]
                oai_messages.append(m)
            else:
                # Anthropic ContentBlock list
                text = ""
                tc = []
                for item in raw:
                    if hasattr(item, "text"):
                        text += item.text
                    elif hasattr(item, "name"):
                        tc.append({
                            "id": item.id,
                            "type": "function",
                            "function": {"name": item.name, "arguments": json.dumps(item.input)},
                        })
                m = {"role": "assistant"}
                if text:
                    m["content"] = text
                if tc:
                    m["tool_calls"] = tc
                oai_messages.append(m)

    response = client.chat.completions.create(model=model, messages=oai_messages, tools=oai_tools)
    choice = response.choices[0]

    text_parts = [choice.message.content] if choice.message.content else []
    tool_calls = []
    json_errors = []
    if choice.message.tool_calls:
        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                # LLM returned truncated/malformed JSON — try to repair
                raw_args = tc.function.arguments
                print(f"  [WARN] Malformed JSON for {tc.function.name}, attempting repair...")
                repaired = _try_repair_json(raw_args)
                if repaired is not None:
                    args = repaired
                    print(f"  [WARN] JSON repaired successfully")
                else:
                    json_errors.append({
                        "id": tc.id, "name": tc.function.name,
                    })
                    print(f"  [WARN] JSON repair failed, will ask LLM to retry")
                    continue
            tool_calls.append({
                "id": tc.id, "name": tc.function.name,
                "arguments": args,
            })
    # Store raw for message history
    raw_content = choice.message
    is_done = choice.finish_reason == "stop"
    return text_parts, tool_calls, raw_content, is_done, json_errors


def _canonical_call_key(name: str, args: dict) -> str:
    """Stable hash key for (tool_name, args). Used by the loop-breaker to detect
    when the LLM is repeating an identical failing tool call.

    SPECIAL CASE for edit_file: weak LLMs (qwen-max) will vary the b64 payload
    slightly on every retry — different leading-whitespace counts, capitalization
    glitches, hallucinated extra spaces inside identifiers. With full-args keying,
    each variation gets its own counter and the loop-breaker never fires. So for
    edit_file we key by file_path only: 'failed to edit the same file 3x in a
    row, regardless of how you varied old_b64'. This makes the hint fire on the
    Nth consecutive failure for a given target file, which is the right signal."""
    try:
        if name == "edit_file" and isinstance(args, dict) and "file_path" in args:
            return f"edit_file::file::{args['file_path']}"
        return name + "::" + json.dumps(args, sort_keys=True, ensure_ascii=False)
    except Exception:
        return name + "::" + repr(args)


def _build_edit_file_hint(args: dict) -> str:
    """When the loop-breaker fires on a repeated edit_file call, fetch the
    target file and use fuzzy matching to find the closest real lines to what
    the LLM was trying to match. Returns those lines verbatim so the LLM has
    a copy-paste reference and can stop hallucinating the content.

    This addresses a failure mode observed with weak LLMs (qwen-max): even
    after a clean read_file, the model regenerates an old_b64 with character-
    level hallucinations ('Import' instead of 'import', 'GPT Config' instead
    of 'GPTConfig'). The auto-repair in edit.py only handles whitespace
    glitches; it can't fix wrong characters. The fix is to embed the EXACT
    target line in the intercept message itself.

    Returns "" on any failure — the basic intercept message is still sent."""
    try:
        import base64
        from difflib import SequenceMatcher
        from pathlib import Path

        file_path = args.get("file_path", "")
        old_b64 = args.get("old_b64", "")
        if not file_path or not old_b64:
            return ""

        # Same tolerant decode as edit.py — strip whitespace, fix padding
        clean = "".join(old_b64.split())
        pad = len(clean) % 4
        if pad:
            clean += "=" * (4 - pad)
        try:
            old_string = base64.b64decode(clean).decode("utf-8")
        except Exception:
            return ""
        if not old_string.strip():
            return ""

        try:
            file_text = Path(file_path).read_text(encoding="utf-8")
        except Exception:
            return ""

        # If old_string spans multiple lines, score against its first non-empty line
        candidate = next((ln for ln in old_string.splitlines() if ln.strip()), old_string)
        candidate = candidate.strip()
        if not candidate:
            return ""

        lines = file_text.splitlines()
        scored = []
        for i, line in enumerate(lines, start=1):
            ratio = SequenceMatcher(None, candidate, line.strip()).ratio()
            scored.append((ratio, i, line))
        scored.sort(key=lambda t: -t[0])
        top = [t for t in scored[:3] if t[0] >= 0.30]
        if not top:
            return ""

        out_lines = [
            "",
            "[FILE-CONTENT-HINT] Your old_b64, decoded, started with:",
            f"  {candidate!r}",
            "",
            f"But that EXACT text is NOT in {file_path}.",
            f"The top {len(top)} closest real lines in the file are shown below.",
            f"For each, the PRECOMPUTED base64 of the line is given — copy it",
            f"VERBATIM into old_b64 (do NOT re-encode the text yourself; weak",
            f"models corrupt the encoding):",
            "",
        ]
        for ratio, lineno, line in top:
            pct = f"{int(ratio * 100):3d}%"
            line_b64 = base64.b64encode(line.encode("utf-8")).decode("ascii")
            out_lines.append(f"  line {lineno} ({pct} match): {line!r}")
            out_lines.append(f"    old_b64 (copy this verbatim): {line_b64}")
        out_lines.append("")
        out_lines.append(
            "Pick the line you actually meant, then COPY its precomputed "
            "old_b64 string above into your next edit_file call. Do NOT "
            "regenerate the base64 yourself — you have already shown you "
            "cannot encode it correctly. Just copy the ASCII string verbatim."
        )
        return "\n".join(out_lines)
    except Exception:
        return ""


def _looks_like_error(result: str) -> bool:
    """Heuristic: did this skill result indicate failure? Used by the loop-breaker.
    We only block repeats of FAILING calls — successful calls are fine to repeat
    (e.g. list_files at different points in time)."""
    head = result[:400].lower()
    # Explicit success marker overrides everything — edit_file prints "[OK] /path"
    # on success and we never want to misclassify that as failure.
    if "[ok]" in head:
        return False
    error_markers = (
        "[error]",      # most skills
        "error:",       # python tracebacks
        "not found",    # edit_file uniqueness fail, file-not-found errors
        "traceback",    # python crashes
        "fail:",        # validate_policy_server outer FAIL: lines
        "import_fail",  # validate_policy_server inner driver marker
        "smoke_fail",   # validate_policy_server smoke driver marker
        "failed",       # generic past-tense reports
    )
    return any(m in head for m in error_markers)


def run_agent(user_message: str, skills: list[dict], model: str = None, provider: str = None):
    """Run the agent ReAct loop."""
    if provider is None:
        provider = _detect_provider()
    if provider is None:
        print("ERROR: No API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or DASHSCOPE_API_KEY.")
        return

    if model is None:
        if provider == "anthropic":
            model = "claude-sonnet-4-6"
        elif os.environ.get("DASHSCOPE_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            model = "qwen-plus"
        else:
            model = "gpt-4o"

    call_llm = _call_anthropic if provider == "anthropic" else _call_openai
    tools = [skill_to_claude_tool(s) for s in skills]
    skill_map = {s["name"]: s for s in skills}

    messages = [{"role": "user", "content": user_message}]

    # Loop-breaker state: track per-call-key consecutive failure counts.
    # If the LLM makes the same (tool, args) call after it's already failed
    # twice, intercept the third attempt with a synthetic error telling it
    # to change something. Reset on first non-error result for that key.
    call_failure_count: dict[str, int] = {}
    LOOP_BREAK_THRESHOLD = 2  # block on the 3rd identical failing attempt

    print(f"\n{'='*60}")
    print(f"  User: {user_message}")
    print(f"  Model: {provider}/{model}")
    print(f"{'='*60}\n")

    max_turns = 15
    for turn in range(max_turns):
        text_parts, tool_calls, raw_content, is_done, json_errors = call_llm(messages, tools, model)

        messages.append({"role": "assistant", "content": raw_content})

        for text in text_parts:
            print(f"  Agent: {text}")

        # Handle malformed JSON — send error feedback so LLM retries
        if json_errors and not tool_calls:
            error_feedback = []
            for je in json_errors:
                error_feedback.append({
                    "type": "tool_result",
                    "tool_use_id": je["id"],
                    "content": f"[ERROR] Your tool call to '{je['name']}' had malformed JSON arguments. "
                               f"Please retry with valid JSON. If the argument is too long (e.g. base64 content), "
                               f"try breaking it into smaller parts or simplifying.",
                    "is_error": True,
                })
            messages.append({"role": "user", "content": error_feedback})
            print(f"  [RETRY] Sent JSON error feedback to LLM (turn {turn+1})")
            continue

        if is_done or (not tool_calls and not json_errors):
            break

        # Execute tool calls
        tool_results = []
        for tc in tool_calls:
            skill = skill_map.get(tc["name"])
            if skill:
                # Loop-breaker: if this exact (tool, args) combo has already
                # failed LOOP_BREAK_THRESHOLD times, intercept instead of
                # executing again. Forces the LLM to actually CHANGE something.
                call_key = _canonical_call_key(tc["name"], tc["arguments"])
                if call_failure_count.get(call_key, 0) >= LOOP_BREAK_THRESHOLD:
                    fail_n = call_failure_count[call_key]
                    intercept_msg = (
                        f"[LOOP-BREAKER] You just made this exact call to "
                        f"'{tc['name']}' with the EXACT SAME arguments {fail_n} "
                        f"times in a row, and it failed each time. STOP retrying "
                        f"with identical arguments — you are stuck in a loop.\n"
                        f"\n"
                        f"REQUIRED next actions:\n"
                        f"  1. Re-read the relevant file (read_file) to confirm "
                        f"the actual current content. The file may not contain "
                        f"what you think it does.\n"
                        f"  2. Change SOMETHING SUBSTANTIVE in your next call: "
                        f"different arguments (re-encoded from a fresh read), "
                        f"a different tool, or a different approach.\n"
                        f"  3. If your previous old_b64 contained a literal "
                        f"newline, that's the LLM token-emission bug — use "
                        f"only single-line content in old_b64. The edit_file "
                        f"skill will auto-strip embedded newlines, but you "
                        f"need to call it with new arguments to trigger that.\n"
                        f"\n"
                        f"This same call will be intercepted again until you "
                        f"change the arguments."
                    )
                    # For edit_file specifically, fetch the real file lines
                    # closest to what the LLM was trying to match. This gives
                    # weak models a copy-paste reference to defeat hallucinations.
                    if tc["name"] == "edit_file":
                        intercept_msg += _build_edit_file_hint(tc["arguments"])
                    print(f"\n  [LOOP-BREAKER] Intercepted repeat of "
                          f"{tc['name']} (failed {fail_n}x already)\n")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc["id"],
                        "content": intercept_msg,
                        "is_error": True,
                    })
                    continue

                print(f"\n  [SKILL] {tc['name']}({json.dumps(tc['arguments'], ensure_ascii=False)})")
                result = execute_skill(skill, tc["arguments"])
                print(f"  [RESULT] {result[:500]}{'...' if len(result) > 500 else ''}\n")

                # Track for loop-breaker: bump count on failure, reset on success
                if _looks_like_error(result):
                    call_failure_count[call_key] = call_failure_count.get(call_key, 0) + 1
                else:
                    call_failure_count.pop(call_key, None)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result,
                })
            else:
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": f"Unknown skill: {tc['name']}",
                    "is_error": True,
                })

        # Include any JSON error feedback alongside successful results
        for je in json_errors:
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": je["id"],
                "content": f"[ERROR] Your tool call to '{je['name']}' had malformed JSON. Please retry with simpler arguments.",
                "is_error": True,
            })

        messages.append({"role": "user", "content": tool_results})

    print(f"\n{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Robot Ops Agent")
    parser.add_argument("message", nargs="?", help="Command to execute (interactive if omitted)")
    parser.add_argument("--model", default=None, help="Model name (auto-detected if omitted)")
    parser.add_argument("--provider", default=None, choices=["anthropic", "openai"], help="LLM provider")
    parser.add_argument("--skills-dir", default=None, help="Path to skills directory")
    args = parser.parse_args()

    # Load skills
    if args.skills_dir:
        skills_dir = Path(args.skills_dir)
    else:
        skills_dir = Path(__file__).parent / "skills"

    # Determine base directory for repo auto-discovery
    base_dir = Path(__file__).parent.parent.parent  # agentRobot/

    # Set AGENTROBOT_ROOT so all skills can use dynamic paths
    os.environ.setdefault("AGENTROBOT_ROOT", str(base_dir))
    print(f"AGENTROBOT_ROOT={os.environ['AGENTROBOT_ROOT']}")

    print(f"Loading skills from {skills_dir}...")
    skills = load_skills(skills_dir)
    print(f"  Loaded {len(skills)} skills from SKILL.md files")

    # Auto-discover policy_server.yaml in sibling repos
    print(f"Discovering policy servers in {base_dir}...")
    discovered = discover_policy_servers(base_dir)
    # Merge: skip if a manually-written skill with same name exists
    existing_names = {s["name"] for s in skills}
    for d in discovered:
        if d["name"] not in existing_names:
            skills.append(d)
            print(f"  [AUTO] {d['name']} (from {d['source']})")
        else:
            print(f"  [SKIP] {d['name']} (manual skill exists)")

    print(f"  Total: {len(skills)} skills: {[s['name'] for s in skills]}")

    if args.message:
        # Single command mode
        run_agent(args.message, skills, model=args.model, provider=args.provider)
    else:
        # Interactive mode
        print("\nRobot Ops Agent — interactive mode (type 'quit' to exit)")
        print("-" * 60)
        while True:
            try:
                user_input = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break
            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            run_agent(user_input, skills, model=args.model, provider=args.provider)


if __name__ == "__main__":
    main()
