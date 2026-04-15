"""Evaluation Registry — model×benchmark compatibility matrix.

Encodes which (model, checkpoint, benchmark) combos work, what config they
need, and whether fine-tuning is required.  Used by EvalPreflightChecker to
gate evaluations before wasting GPU hours.

Design principle:
    Every lesson from a failed 0%-result run should be captured here so that
    the system never silently falls into the same trap twice.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


class Readiness(Enum):
    """How ready is this model×benchmark combo?"""
    READY = "ready"                  # Fine-tuned checkpoint exists, should work
    NEEDS_FINETUNE = "needs_finetune"  # No in-domain checkpoint; must finetune first
    CROSS_DOMAIN = "cross_domain"    # Expect ~0%; useful as generalization baseline
    UNSUPPORTED = "unsupported"      # Known incompatibility (e.g. missing obs key)


@dataclass
class EvalConfig:
    """Configuration for a specific model×benchmark evaluation."""

    # --- Identity ---
    policy_name: str            # e.g. "openvla", "pi0", "spatialvla"
    benchmark: str              # e.g. "libero", "maniskill", "robocasa", "simplerenv"
    readiness: Readiness

    # --- Checkpoint ---
    checkpoint: str = ""        # HF repo or local path
    checkpoint_note: str = ""   # e.g. "LIBERO-spatial finetuned"

    # --- Policy server config ---
    server_script: str = ""     # path relative to AGENTROBOT_ROOT
    server_args: Dict[str, str] = field(default_factory=dict)

    # --- Eval client config ---
    control_mode: str = ""       # e.g. "pd_ee_delta_pose" for ManiSkill
    arm_controller: str = ""     # e.g. "cartesian_pose" for LIBERO/RoboCasa
    action_dim: int = 7          # expected policy output dim
    action_range: tuple = (-1.0, 1.0)  # expected action bounds

    # --- Known issues & mitigations ---
    known_issues: List[str] = field(default_factory=list)
    auto_fixes: Dict[str, str] = field(default_factory=dict)

    # --- Expected performance ---
    expected_success_rate: Optional[float] = None  # from published papers
    expected_source: str = ""  # paper/URL for the expected result

    def summary(self) -> str:
        status = {
            Readiness.READY: "✅ READY",
            Readiness.NEEDS_FINETUNE: "⚠️  NEEDS FINETUNE",
            Readiness.CROSS_DOMAIN: "📊 CROSS-DOMAIN (expect ~0%)",
            Readiness.UNSUPPORTED: "❌ UNSUPPORTED",
        }[self.readiness]
        lines = [
            f"{self.policy_name} × {self.benchmark}: {status}",
            f"  checkpoint: {self.checkpoint or '(none)'}",
        ]
        if self.expected_success_rate is not None:
            lines.append(f"  expected: {self.expected_success_rate:.0%} ({self.expected_source})")
        if self.known_issues:
            for issue in self.known_issues:
                lines.append(f"  ⚠ {issue}")
        return "\n".join(lines)


# ============================================================================
#  REGISTRY — add entries here as we discover what works / doesn't
# ============================================================================

EVAL_REGISTRY: Dict[str, EvalConfig] = {}


def _reg(cfg: EvalConfig):
    key = f"{cfg.policy_name}:{cfg.benchmark}"
    EVAL_REGISTRY[key] = cfg


# ---------------------------------------------------------------------------
#  OpenVLA-OFT × LIBERO (READY — published 97.1%)
#  NOTE: Each LIBERO suite requires its own finetuned checkpoint!
# ---------------------------------------------------------------------------
_OPENVLA_LIBERO_CHECKPOINTS = {
    "libero_spatial": "moojink/openvla-7b-oft-finetuned-libero-spatial",
    "libero_object": "moojink/openvla-7b-oft-finetuned-libero-object",
    "libero_goal": "moojink/openvla-7b-oft-finetuned-libero-goal",
    "libero_10": "moojink/openvla-7b-oft-finetuned-libero-10",
}
# Each suite has its own unnorm_key for action denormalization
_OPENVLA_LIBERO_UNNORM_KEYS = {
    "libero_spatial": "libero_spatial_no_noops",
    "libero_object": "libero_object_no_noops",
    "libero_goal": "libero_goal_no_noops",
    "libero_10": "libero_10_no_noops",
}
for suite, ckpt in _OPENVLA_LIBERO_CHECKPOINTS.items():
    _reg(EvalConfig(
        policy_name="openvla",
        benchmark=f"libero/{suite}",
        readiness=Readiness.READY,
        checkpoint=ckpt,
        checkpoint_note=f"OpenVLA-OFT finetuned on {suite}",
        server_script="openvla/vla-scripts/policy_server.py",
        server_args={
            "--pretrained_checkpoint": ckpt,
            "--unnorm_key": _OPENVLA_LIBERO_UNNORM_KEYS[suite],
        },
        arm_controller="cartesian_pose",
        action_dim=7,
        expected_success_rate=0.97,
        expected_source="OpenVLA-OFT paper Table 1",
    ))

# ---------------------------------------------------------------------------
#  OpenVLA-OFT × ManiSkill (CROSS-DOMAIN)
# ---------------------------------------------------------------------------
_reg(EvalConfig(
    policy_name="openvla",
    benchmark="maniskill",
    readiness=Readiness.CROSS_DOMAIN,
    checkpoint="moojink/openvla-7b-oft-finetuned-libero-spatial",
    checkpoint_note="LIBERO-finetuned, not ManiSkill",
    server_script="openvla/vla-scripts/policy_server.py",
    server_args={"--pretrained_checkpoint": "moojink/openvla-7b-oft-finetuned-libero-spatial"},
    control_mode="pd_ee_delta_pose",
    action_dim=7,
    known_issues=[
        "LIBERO-finetuned model on ManiSkill: different robot, camera, scene → expect 0%",
        "Use --pretrained_checkpoint (not --checkpoint)",
    ],
    expected_success_rate=0.0,
    expected_source="agentRobot v10 smoke test",
))

# ---------------------------------------------------------------------------
#  pi0 × LIBERO (READY — published ~94%)
# ---------------------------------------------------------------------------
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    _reg(EvalConfig(
        policy_name="pi0",
        benchmark=f"libero/{suite}",
        readiness=Readiness.READY,
        checkpoint="lerobot/pi0_libero_finetuned",
        checkpoint_note="pi0 finetuned on LIBERO (v044)",
        server_script="lerobot/policy_server.py",
        server_args={"--checkpoint": "lerobot/pi0_libero_finetuned"},
        arm_controller="cartesian_pose",
        action_dim=7,
        known_issues=["LeRobot LIBERO images require 180° flip (H+W) — handled by policy_server.py"],
        expected_success_rate=0.94,
        expected_source="OpenVLA-OFT paper comparison table",
    ))

# ---------------------------------------------------------------------------
#  pi0.5 × LIBERO (READY — published ~97.5%)
# ---------------------------------------------------------------------------
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    _reg(EvalConfig(
        policy_name="pi0.5",
        benchmark=f"libero/{suite}",
        readiness=Readiness.READY,
        checkpoint="lerobot/pi05_libero_finetuned",
        checkpoint_note="pi0.5 finetuned on LIBERO (LeRobot)",
        server_script="lerobot/policy_server.py",
        server_args={"--checkpoint": "lerobot/pi05_libero_finetuned"},
        arm_controller="cartesian_pose",
        action_dim=7,
        known_issues=["LeRobot LIBERO images require 180° flip (H+W) — handled by policy_server.py"],
        expected_success_rate=0.975,
        expected_source="LeRobot/OpenPI docs",
    ))

# ---------------------------------------------------------------------------
#  Octo × LIBERO (NEEDS_FINETUNE — pretrained gets ~0%)
# ---------------------------------------------------------------------------
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    _reg(EvalConfig(
        policy_name="octo",
        benchmark=f"libero/{suite}",
        readiness=Readiness.NEEDS_FINETUNE,
        checkpoint="hf://rail-berkeley/octo-base-1.5",
        checkpoint_note="Pretrained Octo-Base, NOT finetuned on LIBERO",
        server_script="octo/policy_server.py",
        arm_controller="cartesian_pose",
        action_dim=7,
        known_issues=[
            "Octo pretrained outputs unbounded actions → must clip to [-1,1]",
            "Without LIBERO finetune, expect ~0% (published 75% WITH finetune)",
        ],
        auto_fixes={"unbounded_actions": "np.clip(action, -1.0, 1.0) in policy_server.py"},
        expected_success_rate=0.0,
        expected_source="agentRobot v9 (pretrained, no finetune)",
    ))

# ---------------------------------------------------------------------------
#  Octo × SimplerEnv (READY — pretrained gets ~16-30%)
# ---------------------------------------------------------------------------
_reg(EvalConfig(
    policy_name="octo",
    benchmark="simplerenv",
    readiness=Readiness.READY,
    checkpoint="hf://rail-berkeley/octo-base-1.5",
    checkpoint_note="Pretrained Octo works on SimplerEnv (same OXE pretraining distribution)",
    action_dim=7,
    known_issues=["Octo outputs unbounded actions → must clip"],
    expected_success_rate=0.17,
    expected_source="SimplerEnv paper Table 1 (visual matching)",
))

# ---------------------------------------------------------------------------
#  SpatialVLA × LIBERO (NEEDS_FINETUNE — no LIBERO stats in tokenizer)
# ---------------------------------------------------------------------------
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    _reg(EvalConfig(
        policy_name="spatialvla",
        benchmark=f"libero/{suite}",
        readiness=Readiness.NEEDS_FINETUNE,
        checkpoint="IPEC-COMMUNITY/spatialvla-4b-224-pt",
        checkpoint_note="Pretrained SpatialVLA, no LIBERO-specific statistics",
        server_script="SpatialVLA/policy_server.py",
        server_args={"--checkpoint": "IPEC-COMMUNITY/spatialvla-4b-224-pt", "--unnorm_key": "none"},
        arm_controller="cartesian_pose",
        action_dim=7,
        known_issues=[
            "No LIBERO unnorm stats → must use --unnorm_key none for raw [-1,1] output",
            "bridge_orig stats compress actions to ~0.01 scale → SCALE TOO SMALL",
            "Even with correct normalization, cross-domain → expect ~0% without finetune",
        ],
        auto_fixes={"wrong_unnorm": "Use --unnorm_key none to bypass bridge denormalization"},
        expected_success_rate=0.0,
        expected_source="agentRobot v10 (no LIBERO finetune)",
    ))

# ---------------------------------------------------------------------------
#  SpatialVLA × SimplerEnv (READY — published 71.9% zero-shot Google Robot)
# ---------------------------------------------------------------------------
_reg(EvalConfig(
    policy_name="spatialvla",
    benchmark="simplerenv/google_robot",
    readiness=Readiness.READY,
    checkpoint="IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal",
    checkpoint_note="SpatialVLA finetuned on Fractal (Google Robot)",
    server_script="SpatialVLA/policy_server.py",
    server_args={"--checkpoint": "IPEC-COMMUNITY/spatialvla-4b-224-sft-fractal"},
    action_dim=7,
    expected_success_rate=0.75,
    expected_source="SpatialVLA paper Table 2 (SimplerEnv visual matching, finetuned)",
))

_reg(EvalConfig(
    policy_name="spatialvla",
    benchmark="simplerenv/widowx",
    readiness=Readiness.READY,
    checkpoint="IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge",
    checkpoint_note="SpatialVLA finetuned on BridgeV2 (WidowX)",
    server_script="SpatialVLA/policy_server.py",
    server_args={"--checkpoint": "IPEC-COMMUNITY/spatialvla-4b-224-sft-bridge"},
    action_dim=7,
    expected_success_rate=0.43,
    expected_source="SpatialVLA paper Table 2 (SimplerEnv visual matching, finetuned)",
))

# ---------------------------------------------------------------------------
#  SpatialVLA × ManiSkill (CROSS-DOMAIN)
# ---------------------------------------------------------------------------
_reg(EvalConfig(
    policy_name="spatialvla",
    benchmark="maniskill",
    readiness=Readiness.CROSS_DOMAIN,
    checkpoint="IPEC-COMMUNITY/spatialvla-4b-224-pt",
    server_script="SpatialVLA/policy_server.py",
    server_args={"--checkpoint": "IPEC-COMMUNITY/spatialvla-4b-224-pt", "--unnorm_key": "none"},
    control_mode="pd_ee_delta_pose",
    action_dim=7,
    known_issues=["No ManiSkill-specific training data → expect 0%"],
    expected_success_rate=0.0,
    expected_source="agentRobot v10 smoke test",
))

# ---------------------------------------------------------------------------
#  SmolVLA × LIBERO (NEEDS_FINETUNE — published 87.3% with finetune)
# ---------------------------------------------------------------------------
for suite in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]:
    _reg(EvalConfig(
        policy_name="smolvla",
        benchmark=f"libero/{suite}",
        readiness=Readiness.NEEDS_FINETUNE,
        checkpoint="lerobot/smolvla_base",
        checkpoint_note="SmolVLA base — needs LIBERO finetune (published 87.3% avg)",
        server_script="lerobot/policy_server.py",
        server_args={"--checkpoint": "lerobot/smolvla_base"},
        arm_controller="cartesian_pose",
        action_dim=7,
        known_issues=["Base checkpoint not finetuned on LIBERO → expect ~0%"],
        expected_success_rate=0.0,
        expected_source="agentRobot v9 (base, no finetune)",
    ))

# ---------------------------------------------------------------------------
#  All models × RoboCasa (CROSS-DOMAIN — no model has RoboCasa finetune)
# ---------------------------------------------------------------------------
for model in ["openvla", "pi0", "pi0.5", "octo", "spatialvla", "smolvla"]:
    _reg(EvalConfig(
        policy_name=model,
        benchmark="robocasa",
        readiness=Readiness.CROSS_DOMAIN,
        known_issues=["No VLA model has RoboCasa-finetuned checkpoint → expect 0%"],
        expected_success_rate=0.0,
        expected_source="agentRobot v9 cross-domain baseline",
    ))


# ============================================================================
#  Lookup helpers
# ============================================================================

def lookup(policy_name: str, benchmark: str) -> Optional[EvalConfig]:
    """Look up eval config by policy and benchmark name.

    Tries exact match first, then prefix match (e.g. "libero" matches
    "libero/libero_spatial").
    """
    key = f"{policy_name}:{benchmark}"
    if key in EVAL_REGISTRY:
        return EVAL_REGISTRY[key]
    # Try prefix match for benchmark suites
    for k, v in EVAL_REGISTRY.items():
        if k.startswith(f"{policy_name}:") and benchmark in k:
            return v
    return None


def list_ready(benchmark: str = "") -> List[EvalConfig]:
    """List all READY configs, optionally filtered by benchmark."""
    results = []
    for cfg in EVAL_REGISTRY.values():
        if cfg.readiness == Readiness.READY:
            if not benchmark or benchmark in cfg.benchmark:
                results.append(cfg)
    return results


def print_matrix():
    """Print the full compatibility matrix."""
    benchmarks = sorted(set(c.benchmark for c in EVAL_REGISTRY.values()))
    policies = sorted(set(c.policy_name for c in EVAL_REGISTRY.values()))

    symbols = {
        Readiness.READY: "✅",
        Readiness.NEEDS_FINETUNE: "⚠️ ",
        Readiness.CROSS_DOMAIN: "📊",
        Readiness.UNSUPPORTED: "❌",
    }

    print(f"\n{'Model':<15}", end="")
    for b in benchmarks:
        short = b.split("/")[-1] if "/" in b else b
        print(f"{short:<18}", end="")
    print()
    print("-" * (15 + 18 * len(benchmarks)))

    for p in policies:
        print(f"{p:<15}", end="")
        for b in benchmarks:
            key = f"{p}:{b}"
            cfg = EVAL_REGISTRY.get(key)
            if cfg:
                sym = symbols[cfg.readiness]
                rate = f"{cfg.expected_success_rate:.0%}" if cfg.expected_success_rate is not None else "?"
                print(f"{sym} {rate:<14}", end="")
            else:
                print(f"{'--':<18}", end="")
        print()
    print()
    print("✅ = ready  ⚠️  = needs finetune  📊 = cross-domain (~0%)  ❌ = unsupported")
