---
name: edit_file
description: "Targeted string replacement in a file. Use this for SMALL FIXES (one line, a few lines, a single import) instead of write_file — the LLM only needs to encode the old + new substring, not the whole file. Errors loudly if old_string is not found or matches more than once. Atomic write."
version: 1.0.0
category: util
parameters:
  file_path:
    type: string
    description: "Absolute path to the file to edit"
    required: true
  old_b64:
    type: string
    description: "Base64-encoded EXACT substring to find. Must match whitespace/indentation precisely. Must be unique in the file unless replace_all is set."
    required: true
  new_b64:
    type: string
    description: "Base64-encoded replacement substring. Must differ from old_b64."
    required: true
  replace_all:
    type: string
    description: "If 'true', replace ALL occurrences. Default 'false' (must be unique)."
    default: "false"
requires:
  bins: [python3]
timeout: 15
command_template: |
  python3 $AGENTROBOT_ROOT/agentic/robot_agent/skills/edit_file/edit.py "{file_path}" --old-b64 "{old_b64}" --new-b64 "{new_b64}" $( [ "{replace_all}" = "true" ] && echo --replace-all )
---

# Edit File

The robust patch primitive. Use this for any change small enough that the
old + new strings together fit in a few hundred bytes. The agent only needs
to base64-encode the substring being replaced — not the whole file — which
makes it dramatically more reliable than `write_file` for small fixes.

## When to use
- Fixing one or two import lines (`from gpt import` → `from pkg.gpt import`)
- Renaming a class, method, variable
- Patching a single function body
- Adding a `_validate_only` shortcut to an existing `__init__`

## When NOT to use
- Creating a new file from scratch (use `write_file`)
- Rewriting >50% of a file (use `write_file`)

## Workflow with the validate loop
```
read_file(...)                     # see exact text and indentation
edit_file(file_path=..., old_b64=..., new_b64=...)
validate_policy_server(mode=smoke)
  ↳ FAIL → read_file again, refine the patch, edit_file again
  ↳ PASS → done
```

## Failure modes (and how the LLM should react)
- `old_string NOT FOUND` → re-read the file. Whitespace/indentation must match
  EXACTLY. Don't guess — copy from a `read_file` output.
- `matches N places — not unique` → include more surrounding context in
  `old_string` (e.g. add the line above and below) until it's unique.
- `old_string and new_string are identical` → you have a logic bug, no edit
  needed.

## Why base64?
The agent's command_template runs through `bash -c`, which mangles raw
newlines, quotes, and shell metacharacters. Base64 keeps the strings exactly
as the LLM intends.
