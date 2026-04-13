#!/usr/bin/env python3
"""
Smart adapter generator — reads repo source code to generate a working policy_server.py.
Unlike the template approach, this:
1. Scans for inference/demo scripts to find usage patterns
2. Generates framework-specific model loading code
3. Produces a complete, runnable adapter
"""

import argparse
import re
import sys
from pathlib import Path


def find_inference_patterns(repo_path: Path) -> dict:
    """Scan repo for inference patterns."""
    patterns = {
        "load_pattern": None,
        "predict_pattern": None,
        "obs_keys": [],
        "has_from_pretrained": False,
        "has_load_pretrained": False,
        "has_hydra": False,
        "has_checkpoint_arg": False,
        "has_algo_factory": False,
        "has_deserialize": False,
        "demo_files": [],
    }

    # Find demo/inference scripts
    for pattern in ["**/demo*.py", "**/eval*.py", "**/infer*.py", "**/predict*.py", "**/run*.py"]:
        for f in repo_path.glob(pattern):
            if ".venv" not in str(f) and "__pycache__" not in str(f):
                patterns["demo_files"].append(str(f.relative_to(repo_path)))
                try:
                    code = f.read_text(errors="ignore")
                    if "from_pretrained" in code:
                        patterns["has_from_pretrained"] = True
                    if "load_pretrained" in code:
                        patterns["has_load_pretrained"] = True
                    if "@hydra" in code or "hydra.main" in code or "OmegaConf" in code:
                        patterns["has_hydra"] = True
                    if "--checkpoint" in code or "--ckpt" in code:
                        patterns["has_checkpoint_arg"] = True
                    # Find observation key patterns
                    for m in re.finditer(r'["\'](?:agentview_image|robot0_\w+|image_\w+|primary_image)["\']', code):
                        key = m.group().strip("'\"")
                        if key not in patterns["obs_keys"]:
                            patterns["obs_keys"].append(key)
                except Exception:
                    pass

    # Scan model source for patterns
    for f in repo_path.rglob("*.py"):
        if ".venv" in str(f) or "__pycache__" in str(f):
            continue
        try:
            code = f.read_text(errors="ignore")
            if "def predict(" in code or "def predict_action(" in code:
                patterns["predict_pattern"] = "predict"
            if "def get_action(" in code:
                patterns["predict_pattern"] = "get_action"
            if "def sample_actions(" in code:
                patterns["predict_pattern"] = "sample_actions"
            if "algo_factory" in code:
                patterns["has_algo_factory"] = True
            if "def deserialize(" in code:
                patterns["has_deserialize"] = True
        except Exception:
            pass

    return patterns


def generate_torch_adapter(repo_name, model_class, model_module, checkpoint, action_dim, patterns):
    """Generate PyTorch adapter."""
    load_code = f"""
        import torch
        from {model_module} import {model_class}
"""
    if patterns["has_algo_factory"] and patterns["has_deserialize"]:
        pkg = model_module.split('.')[0]  # top-level package (e.g. robomimic)
        load_code += f"""
        # robomimic-style factory loading (algo_factory + deserialize)
        import {pkg}.utils.file_utils as FileUtils
        from {pkg}.algo import algo_factory
        import {pkg}.utils.obs_utils as ObsUtils

        ckpt_path = kwargs.get("checkpoint", "{checkpoint}")
        ckpt_dict = FileUtils.load_dict_from_checkpoint(ckpt_path)
        algo_name, _ = FileUtils.algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
        config, _ = FileUtils.config_from_checkpoint(algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=True)
        ObsUtils.initialize_obs_utils_with_config(config)
        shape_meta = ckpt_dict["shape_metadata"]
        self.model = algo_factory(algo_name, config, obs_key_shapes=shape_meta["all_shapes"],
                                   ac_dim=shape_meta["ac_dim"], device="cuda")
        self.model.deserialize(ckpt_dict["model"])
        self.model.set_eval()
"""
    elif patterns["has_hydra"]:
        load_code += f"""
        # Hydra/OmegaConf config-based loading
        from omegaconf import OmegaConf
        import hydra
        ckpt = torch.load(kwargs.get("checkpoint", "{checkpoint}"), map_location="cpu")
        cfg = ckpt.get("cfg", ckpt.get("config", {{}}))
        if isinstance(cfg, dict):
            cfg = OmegaConf.create(cfg)
        self.model = hydra.utils.instantiate(cfg.policy) if hasattr(cfg, "policy") else {model_class}(**cfg)
        if "state_dict" in ckpt:
            self.model.load_state_dict(ckpt["state_dict"])
        elif "model" in ckpt:
            self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.cuda()
"""
    elif patterns["has_from_pretrained"]:
        load_code += f"""
        self.model = {model_class}.from_pretrained(kwargs.get("checkpoint", "{checkpoint}"))
        self.model.eval()
        self.model.cuda()
"""
    else:
        load_code += f"""
        ckpt_path = kwargs.get("checkpoint", "{checkpoint}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt
        self.model = {model_class}()
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.cuda()
"""

    predict = patterns.get("predict_pattern", "predict")

    return f'''#!/usr/bin/env python3
"""
Policy server adapter for {repo_name}.
Generated by Robot Ops Agent (smart wrap_policy).

Usage:
    python policy_server.py --port 18800 --checkpoint /path/to/checkpoint.ckpt
"""

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "policy_websocket" / "src"))

from policy_websocket import BasePolicy, WebsocketPolicyServer


class {repo_name.title().replace("_","")}Policy(BasePolicy):
    """Wraps {repo_name} model as a BasePolicy."""

    def __init__(self, **kwargs):
{load_code}
        self._action_dim = {action_dim}
        print(f"[{repo_name}] Model loaded on {{next(self.model.parameters()).device}}")

    def infer(self, obs: Dict) -> Dict:
        if not self._has_images(obs):
            self._action_dim = int(obs.get("action_dim", self._action_dim))
            return {{"actions": np.zeros(self._action_dim, dtype=np.float64)}}

        model_input = self._preprocess(obs)
        with torch.no_grad():
            raw_action = self.model.{predict}(model_input)
        if isinstance(raw_action, torch.Tensor):
            action = raw_action.cpu().numpy().flatten()
        elif isinstance(raw_action, dict):
            action = raw_action.get("action", raw_action.get("actions", np.zeros(self._action_dim)))
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy().flatten()
        else:
            action = np.asarray(raw_action).flatten()
        return {{"actions": action[:self._action_dim]}}

    def reset(self):
        if hasattr(self.model, "reset"):
            self.model.reset()

    @staticmethod
    def _has_images(obs: Dict) -> bool:
        image_keys = ("agentview_image", "robot0_eye_in_hand_image",
                      "robot0_agentview_left_image", "primary_image", "image_primary")
        return any(k in obs and obs[k] is not None for k in image_keys)

    def _preprocess(self, obs: Dict) -> Dict:
        """Convert raw obs to model input tensors."""
        processed = {{}}
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v).float()
                if t.ndim == 3 and t.shape[-1] == 3:  # HWC image -> CHW
                    t = t.permute(2, 0, 1) / 255.0
                processed[k] = t.unsqueeze(0).cuda()  # add batch dim
            elif isinstance(v, (int, float)):
                processed[k] = v
            else:
                processed[k] = v
        return processed


def main():
    parser = argparse.ArgumentParser(description="{repo_name} Policy Server")
    parser.add_argument("--port", type=int, default=18800)
    parser.add_argument("--checkpoint", type=str, default="{checkpoint}")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    policy = {repo_name.title().replace("_","")}Policy(checkpoint=args.checkpoint)
    server = WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port,
        metadata={{"model": "{repo_name}", "action_dim": {action_dim}}},
    )

    print(f"{repo_name} policy server on ws://{{args.host}}:{{args.port}}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
'''


def generate_jax_adapter(repo_name, model_class, model_module, checkpoint, action_dim, patterns):
    """Generate JAX adapter."""
    predict = patterns.get("predict_pattern", "sample_actions")

    return f'''#!/usr/bin/env python3
"""
Policy server adapter for {repo_name} (JAX).
Generated by Robot Ops Agent (smart wrap_policy).
"""

import argparse
import contextlib
import sys
from pathlib import Path
from typing import Dict

import jax
import jax.numpy as jnp
import numpy as np

_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "policy_websocket" / "src"))

from {model_module} import {model_class}
from policy_websocket import BasePolicy, WebsocketPolicyServer


class {repo_name.title().replace("_","")}Policy(BasePolicy):
    """Wraps {repo_name} model as a BasePolicy."""

    def __init__(self, **kwargs):
        ckpt = kwargs.get("checkpoint", "{checkpoint}")
        print(f"Loading {{ckpt}}...")
        self.model = {model_class}.load_pretrained(ckpt)
        self._rng = jax.random.PRNGKey(0)
        self._task = None
        self._action_dim = {action_dim}
        print(f"[{repo_name}] Loaded. Devices: {{jax.devices()}}")

    def infer(self, obs: Dict) -> Dict:
        if not self._has_images(obs):
            self._action_dim = int(obs.get("action_dim", self._action_dim))
            prompt = obs.get("prompt", obs.get("task_description", ""))
            if prompt and hasattr(self.model, "create_tasks"):
                self._task = self.model.create_tasks(texts=[prompt])
            return {{"actions": np.zeros(self._action_dim, dtype=np.float64)}}

        octo_obs = self._preprocess(obs)
        if self._task is None and hasattr(self.model, "create_tasks"):
            self._task = self.model.create_tasks(texts=["complete the task"])

        self._rng, key = jax.random.split(self._rng)
        actions = self.model.{predict}(octo_obs, self._task, rng=key)
        return {{"actions": np.asarray(actions[0, 0]).flatten()}}

    def reset(self):
        self._rng = jax.random.PRNGKey(0)
        self._task = None

    @staticmethod
    def _has_images(obs: Dict) -> bool:
        image_keys = ("image_primary", "primary_image", "agentview_image",
                      "robot0_agentview_left_image")
        return any(k in obs and obs[k] is not None for k in image_keys)

    def _preprocess(self, obs: Dict) -> Dict:
        img = None
        for k in ("image_primary", "primary_image", "agentview_image"):
            if k in obs and obs[k] is not None:
                img = np.asarray(obs[k], dtype=np.uint8)
                break
        if img is None:
            raise ValueError(f"No image in obs: {{list(obs.keys())}}")
        return {{
            "image_primary": img[np.newaxis, np.newaxis],
            "timestep_pad_mask": np.ones((1, 1), dtype=bool),
        }}


def main():
    parser = argparse.ArgumentParser(description="{repo_name} Policy Server")
    parser.add_argument("--port", type=int, default=18800)
    parser.add_argument("--checkpoint", type=str, default="{checkpoint}")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    policy = {repo_name.title().replace("_","")}Policy(checkpoint=args.checkpoint)
    server = WebsocketPolicyServer(
        policy=policy, host=args.host, port=args.port,
        metadata={{"model": "{repo_name}", "action_dim": {action_dim}}},
    )

    print(f"{repo_name} policy server on ws://{{args.host}}:{{args.port}}")
    with contextlib.suppress(KeyboardInterrupt):
        server.serve_forever()


if __name__ == "__main__":
    main()
'''


def main():
    parser = argparse.ArgumentParser(description="Smart policy_server.py generator")
    parser.add_argument("repo_path")
    parser.add_argument("model_class")
    parser.add_argument("model_module")
    parser.add_argument("checkpoint", nargs="?", default="")
    parser.add_argument("action_dim", nargs="?", type=int, default=7)
    parser.add_argument("framework", nargs="?", default="torch")
    args = parser.parse_args()

    repo_path = Path(args.repo_path)
    repo_name = repo_path.name

    print(f"=== Smart Adapter Generator ===")
    print(f"Repo: {repo_name}, Framework: {args.framework}")

    # Scan repo for patterns
    patterns = find_inference_patterns(repo_path)
    print(f"Demo files found: {patterns['demo_files'][:5]}")
    print(f"Predict method: {patterns['predict_pattern']}")
    print(f"Hydra config: {patterns['has_hydra']}")
    print(f"from_pretrained: {patterns['has_from_pretrained']}")

    # Generate adapter
    if args.framework == "jax":
        code = generate_jax_adapter(repo_name, args.model_class, args.model_module,
                                     args.checkpoint, args.action_dim, patterns)
    else:
        code = generate_torch_adapter(repo_name, args.model_class, args.model_module,
                                       args.checkpoint, args.action_dim, patterns)

    output = repo_path / "policy_server.py"
    output.write_text(code)
    print(f"\n=== Generated: {output} ===")
    print(f"Lines: {len(code.splitlines())}")
    print(f"Framework: {args.framework}")
    print(f"Model: {args.model_class} from {args.model_module}")
    print(f"Predict method: {patterns['predict_pattern'] or 'predict'}")

    # Also write config
    config = repo_path / "_policy_adapter_config.yaml"
    config.write_text(
        f'repo_name: "{repo_name}"\n'
        f'model_class: "{args.model_class}"\n'
        f'model_module: "{args.model_module}"\n'
        f'checkpoint: "{args.checkpoint}"\n'
        f'action_dim: {args.action_dim}\n'
        f'framework: "{args.framework}"\n'
        f'predict_method: "{patterns["predict_pattern"] or "predict"}"\n'
        f'hydra_config: {patterns["has_hydra"]}\n'
    )


if __name__ == "__main__":
    main()
