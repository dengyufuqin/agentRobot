---
name: validate_dataloader
description: "Before launching a multi-hour finetune, prove the dataloader factory actually works: imports, instantiates, yields batches, tensors are finite, and expected keys are present. Fails fast with a structured JSON report so the LLM can iterate on the dataloader code without wasting GPU hours."
version: 1.0.0
category: validation
parameters:
  factory_module:
    type: string
    description: "Dotted import path OR absolute .py file path of the module containing the dataloader factory"
    required: true
  factory_func:
    type: string
    description: "Function name inside the module (called with factory_kwargs)"
    default: "make_dataloader"
  factory_kwargs:
    type: string
    description: "JSON-encoded dict of kwargs to pass to the factory (e.g. '{\"repo_id\": \"lerobot/libero_spatial\", \"batch_size\": 2}')"
    default: "{}"
  expected_keys:
    type: string
    description: "Comma-separated keys that every batch must contain (e.g. 'image,state,action')"
    required: false
  num_batches:
    type: integer
    description: "How many batches to pull and inspect"
    default: 2
  pythonpath:
    type: string
    description: "Colon-separated extra PYTHONPATH entries"
    required: false
  out:
    type: string
    description: "Optional absolute path to dump the JSON report (otherwise prints to stdout)"
    required: false
  python:
    type: string
    description: "Python interpreter (defaults to repo's .venv/bin/python3 or system python3)"
    required: false
requires:
  bins: [python3]
timeout: 180
command_template: |
  VALIDATOR="$AGENTROBOT_ROOT/agentic/robot_agent/skills/validate_dataloader/validate_dataloader.py"
  PY="{python}"
  if [ -z "$PY" ]; then PY="python3"; fi

  OUT_ARG=""
  if [ -n "{out}" ]; then OUT_ARG="--out {out}"; fi
  KEYS_ARG=""
  if [ -n "{expected_keys}" ]; then KEYS_ARG="--expected-keys {expected_keys}"; fi
  PP_ARG=""
  if [ -n "{pythonpath}" ]; then PP_ARG="--pythonpath {pythonpath}"; fi

  "$PY" "$VALIDATOR" \
    --factory-module "{factory_module}" \
    --factory-func "{factory_func}" \
    --factory-kwargs '{factory_kwargs}' \
    --num-batches {num_batches} \
    $KEYS_ARG $PP_ARG $OUT_ARG
---

# Validate Dataloader

Pulls 1-2 batches from a user-supplied dataloader factory and reports structure.

## When to use

Before kicking off `finetune` on a new dataset / new dataloader — a 4-hour job that crashes in the first 30s because the dataloader has a wrong key is the exact failure this prevents.

## Exit codes

| Exit | Meaning |
|---|---|
| 0 | all checks pass — safe to finetune |
| 1 | factory not importable (bad path, syntax error, missing dep) |
| 2 | factory raised on instantiation (dataset not found, config bad) |
| 3 | first batch raised / empty dataset |
| 4 | shape/dtype wrong, NaN/Inf detected, or expected keys missing |

## Report shape

```json
{
  "factory": "my_dataset:make_dataloader",
  "factory_kwargs": {"batch_size": 2},
  "stages": {
    "import":      {"ok": true, "seconds": 0.4},
    "instantiate": {"ok": true, "seconds": 1.2, "type": "DataLoader"},
    "iterate":     {"ok": true, "seconds": 2.1, "n_batches": 2},
    "verify":      {"ok": true, "problems": []}
  },
  "sample_batches": [
    {"image": {"shape": [2, 3, 224, 224], "dtype": "torch.float32", "finite": true, "device": "cpu"},
     "state": {"shape": [2, 8],           "dtype": "torch.float32", "finite": true, "device": "cpu"},
     "action":{"shape": [2, 7],           "dtype": "torch.float32", "finite": true, "device": "cpu"}}
  ]
}
```

## Usage

```
validate_dataloader(
  factory_module="/path/to/repo/my_dataset.py",
  factory_func="make_dataloader",
  factory_kwargs='{"repo_id": "lerobot/libero_spatial", "batch_size": 2}',
  expected_keys="observation.image,observation.state,action",
  python="/path/to/repo/.venv/bin/python3",
)
```
