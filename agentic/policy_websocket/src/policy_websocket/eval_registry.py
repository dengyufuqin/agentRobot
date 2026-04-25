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


# ---------------------------------------------------------------------------
#  ManiSkill READY entries — locally finetuned checkpoints in agent_robot datasets
#  Published baselines come from the RPD paper (arXiv 2503.05833) where
#  Juelg/OpenVLA-7B-finetuned-maniskill and Juelg/Octo-base-1.5-finetuned-maniskill
#  were released. pi0.5 ManiSkill checkpoint is our own finetune.
#  These override the earlier CROSS_DOMAIN entries (last write wins).
# ---------------------------------------------------------------------------
_LOCAL_CKPT_ROOT = "/mnt/vast/workspaces/JAIF/dy/datasets/agent_robot/checkpoints"

_reg(EvalConfig(
    policy_name="octo",
    benchmark="maniskill",
    readiness=Readiness.READY,
    checkpoint=f"{_LOCAL_CKPT_ROOT}/octo-maniskill",
    checkpoint_note="Octo-base-1.5 finetuned on ManiSkill (RPD paper). Top-level dir (has config.json); step weights live under 60000/default/.",
    server_script="octo/policy_server.py",
    server_args={"--checkpoint": f"{_LOCAL_CKPT_ROOT}/octo-maniskill"},
    control_mode="pd_ee_delta_pose",
    action_dim=7,
    known_issues=[
        "Path MUST be top-level ckpt dir, NOT /60000 — config.json lives at top, weights at 60000/default/.",
        "Outputs are unbounded; clipped to [-1,1] in policy_server.py.",
    ],
    expected_success_rate=0.52,
    expected_source="RPD paper (2503.05833) Table 2",
))

_reg(EvalConfig(
    policy_name="openvla",
    benchmark="maniskill",
    readiness=Readiness.READY,
    checkpoint=f"{_LOCAL_CKPT_ROOT}/openvla-maniskill",
    checkpoint_note="OpenVLA-7B finetuned on ManiSkill (Juelg, RPD paper).",
    server_script="openvla/vla-scripts/policy_server.py",
    server_args={
        "--pretrained_checkpoint": f"{_LOCAL_CKPT_ROOT}/openvla-maniskill",
        "--unnorm_key": "maniskill_human:7.0.0",
        "--num_images_in_input": "1",
        "--no_proprio": "",
    },
    control_mode="pd_ee_delta_pose",
    action_dim=7,
    known_issues=[
        "Older modeling_prismatic.py without self.llm_dim → policy_server.py falls back to config.text_config.hidden_size.",
        "Base OpenVLA (no OFT action_head/proprio) → server auto-falls back to discrete-token decoding.",
        "unnorm_key must be 'maniskill_human:7.0.0' (not libero_spatial_no_noops default).",
    ],
    expected_success_rate=0.65,
    expected_source="RPD paper (2503.05833) Table 2",
))

_reg(EvalConfig(
    policy_name="pi0.5",
    benchmark="maniskill",
    readiness=Readiness.READY,
    checkpoint=f"{_LOCAL_CKPT_ROOT}/pi05-maniskill",
    checkpoint_note="pi0.5 finetuned on ManiSkill (local, safetensors format).",
    server_script="lerobot/policy_server.py",
    server_args={"--checkpoint": f"{_LOCAL_CKPT_ROOT}/pi05-maniskill"},
    control_mode="pd_ee_delta_pose",
    action_dim=7,
    known_issues=[
        "ManiSkill obs must include robot0_joint_pos for DROID-format conversion (patched in ManiSkill/mani_skill/utils/run_utils.py).",
    ],
    expected_success_rate=0.80,
    expected_source="Internal target (matches pi0.5 LIBERO published 97.5%)",
))


# ---------------------------------------------------------------------------
#  RoboTwin READY entries — community-released fine-tuned checkpoints on
#  RoboTwin 2.0. The RoboTwin authors did NOT publish per-task ckpts; the
#  ones below are third-party HuggingFace uploads. Paper leaderboard for
#  reference: robotwin-platform.github.io/leaderboard.
# ---------------------------------------------------------------------------
_ROBOTWIN_CKPT_ROOT = f"{_LOCAL_CKPT_ROOT}/robotwin_ckpts"

# OpenVLA-OFT × robotwin:beat_block_hammer — Haozhan72/ALOHA 14D bimanual
_reg(EvalConfig(
    policy_name="openvla",
    benchmark="robotwin:beat_block_hammer",
    readiness=Readiness.READY,
    checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/robotwin2_beat_block_hammer_seed1k_sft_aloha_25chunks_10k",
    checkpoint_note="OpenVLA-OFT 7B finetuned on RoboTwin 2.0 beat_block_hammer (Haozhan72, seed1k SFT, 10k steps). 14D bimanual ALOHA action.",
    server_script="openvla/vla-scripts/policy_server.py",
    server_args={
        "--pretrained_checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/robotwin2_beat_block_hammer_seed1k_sft_aloha_25chunks_10k",
        "--unnorm_key": "beat_block_hammer_1k",
        "--aloha": "",
    },
    arm_controller="joint_pos",
    action_dim=14,
    action_range=(-3.15, 3.15),
    known_issues=[
        "14D bimanual ALOHA joint-pos action. --aloha preset sets num_images_in_input=3, "
        "use_film=True, use_proprio=True (14D), execute_steps=25, skips gripper normalize+invert, "
        "and triggers ALOHA_CONSTANTS in prismatic/vla/constants.py.",
        "3-camera obs: head + left wrist + right wrist (sourced from RoboTwin native keys).",
        "unnorm_key is 'beat_block_hammer_1k' (in dataset_statistics.json).",
    ],
    expected_success_rate=0.30,
    expected_source="RoboTwin 2.0 leaderboard OpenVLA-oft pending; paper OpenVLA baseline ~30% (Easy)",
))

# OpenVLA-OFT × robotwin:stack_blocks_two — Haozhan72/ALOHA 14D bimanual, 40k steps
_reg(EvalConfig(
    policy_name="openvla",
    benchmark="robotwin:stack_blocks_two",
    readiness=Readiness.READY,
    checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/robotwin2_stack_blocks_two_seed1k_sft_aloha_25chunks_40k",
    checkpoint_note="OpenVLA-OFT 7B finetuned on RoboTwin 2.0 stack_blocks_two (Haozhan72, seed1k SFT, 40k steps). 14D bimanual ALOHA action.",
    server_script="openvla/vla-scripts/policy_server.py",
    server_args={
        "--pretrained_checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/robotwin2_stack_blocks_two_seed1k_sft_aloha_25chunks_40k",
        "--unnorm_key": "stack_blocks_two_1k",
        "--aloha": "",
    },
    arm_controller="joint_pos",
    action_dim=14,
    action_range=(-3.15, 3.15),
    known_issues=[
        "Same --aloha preset as beat_block_hammer: 3-camera + 14D proprio + bimanual action-pp skip.",
        "unnorm_key is 'stack_blocks_two_1k' (in dataset_statistics.json).",
        "Trained 4× longer than beat_block_hammer ckpt (40k vs 10k steps).",
    ],
    expected_success_rate=0.30,
    expected_source="RoboTwin 2.0 leaderboard OpenVLA-oft pending; paper OpenVLA baseline ~30% (Easy)",
))

# pi0 (lerobot) × robotwin:beat_block_hammer — nextbig multi-task (5 tasks, 30fps)
_reg(EvalConfig(
    policy_name="pi0",
    benchmark="robotwin:beat_block_hammer",
    readiness=Readiness.READY,
    checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/pi0-robotwin-30fps-tasks5",
    checkpoint_note="pi0 lerobot-format multi-task finetune on 5 RoboTwin tasks at 30fps (nextbig). max_state_dim=32, max_action_dim=32 (pi0 padding).",
    server_script="lerobot/policy_server.py",
    server_args={"--checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/pi0-robotwin-30fps-tasks5"},
    arm_controller="joint_pos",
    action_dim=14,
    action_range=(-3.15, 3.15),
    known_issues=[
        "Task list in ckpt not published; beat_block_hammer may or may not be one of the 5 trained tasks.",
        "lerobot pi0 pads to 32D internally; RoboTwin adapter must provide 14D proprio.",
    ],
    expected_success_rate=0.46,
    expected_source="RoboTwin 2.0 leaderboard Pi0 Easy avg 46% (aspirational — if task is in the trained subset)",
))

# RDT-1B × robotwin:click_bell — OPT12 per-task fine-tune
_reg(EvalConfig(
    policy_name="rdt",
    benchmark="robotwin:click_bell",
    readiness=Readiness.READY,
    checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/robotwin2_click_bell_RDT_step15k",
    checkpoint_note="RDT-1B finetuned on RoboTwin 2.0 click_bell (OPT12, 15k steps). DeepSpeed ZeRO ckpt layout: pytorch_model/mp_rank_00_model_states.pt.",
    server_script="RDT/policy_server.py",
    server_args={
        "--checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/robotwin2_click_bell_RDT_step15k",
        "--robotwin": "",
    },
    arm_controller="joint_pos",
    action_dim=14,
    action_range=(-3.15, 3.15),
    known_issues=[
        "--robotwin preset: agilex_model factory + 3-camera (cam_high/right_wrist/left_wrist) + 14D state (left 6+1 + right 6+1).",
        "Local T5/SigLIP from robotwin_ckpts/ (no HF download at runtime).",
        "Server auto-resolves pytorch_model/mp_rank_00_model_states.pt inside ckpt dir.",
    ],
    expected_success_rate=0.35,
    expected_source="RoboTwin 2.0 leaderboard RDT Easy avg 35%",
))

# RDT-1B × robotwin:place_empty_cup — OPT12 per-task fine-tune (12k steps)
_reg(EvalConfig(
    policy_name="rdt",
    benchmark="robotwin:place_empty_cup",
    readiness=Readiness.READY,
    checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/robotwin2_place_empty_cup_RDT_step12k",
    checkpoint_note="RDT-1B finetuned on RoboTwin 2.0 place_empty_cup (OPT12, 12k steps).",
    server_script="RDT/policy_server.py",
    server_args={
        "--checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/robotwin2_place_empty_cup_RDT_step12k",
        "--robotwin": "",
    },
    arm_controller="joint_pos",
    action_dim=14,
    action_range=(-3.15, 3.15),
    expected_success_rate=0.35,
    expected_source="RoboTwin 2.0 leaderboard RDT Easy avg 35%",
))


# ---------------------------------------------------------------------------
#  OpenVLA-OFT × RoboTwin — additional per-task entries (Haozhan72, --aloha preset)
# ---------------------------------------------------------------------------
def _openvla_oft_robotwin(task: str, ckpt_dir_name: str, unnorm_key: str,
                          steps_desc: str = "seed1k SFT"):
    _reg(EvalConfig(
        policy_name="openvla",
        benchmark=f"robotwin:{task}",
        readiness=Readiness.READY,
        checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/{ckpt_dir_name}",
        checkpoint_note=f"OpenVLA-OFT 7B finetuned on RoboTwin 2.0 {task} (Haozhan72, {steps_desc}). 14D bimanual ALOHA action.",
        server_script="openvla/vla-scripts/policy_server.py",
        server_args={
            "--pretrained_checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/{ckpt_dir_name}",
            "--unnorm_key": unnorm_key,
            "--aloha": "",
        },
        arm_controller="joint_pos",
        action_dim=14,
        action_range=(-3.15, 3.15),
        known_issues=[
            "--aloha preset: 3-camera + 14D proprio + bimanual action-pp skip.",
            f"unnorm_key is '{unnorm_key}' (in dataset_statistics.json).",
        ],
        expected_success_rate=0.30,
        expected_source="RoboTwin 2.0 leaderboard OpenVLA-oft pending; paper OpenVLA baseline ~30% (Easy)",
    ))

_openvla_oft_robotwin("stack_bowls_two",
                      "robotwin2_stack_bowls_two_seed1k_sft_aloha_25chunks_40k",
                      "stack_bowls_two_1k", "seed1k SFT, 40k steps")
_openvla_oft_robotwin("place_empty_cup",
                      "robotwin2_place_empty_cup_seed1k_sft_aloha_25chunks_10k",
                      "place_empty_cup_1k", "seed1k SFT, 10k steps")
_openvla_oft_robotwin("place_container_plate",
                      "robotwin2_place_container_plate_seed1k_sft_aloha_25chunks_10k",
                      "place_container_plate_1k", "seed1k SFT, 10k steps")
_openvla_oft_robotwin("pick_dual_bottles",
                      "pick_dual_bottles_seed1k_sft_aloha_25chunks_10k",
                      "pick_dual_bottles_1k", "seed1k SFT, 10k steps")
_openvla_oft_robotwin("lift_pot",
                      "twin2_lift_pot_seed1k_sft_aloha_25chunks_20k",
                      "lift_pot_1k", "seed1k SFT, 20k steps")
_openvla_oft_robotwin("handover_block",
                      "twin2_handover_block_seed1k_sft_aloha_25chunks",
                      "handover_block_1k", "seed1k SFT")


# ---------------------------------------------------------------------------
#  ACT × RoboTwin — Avada11/AgileX bimanual ALOHA, per-task ckpts
#  reference: Avada11/RoboTwin-Model-AgileX-ACT (HF), ~83.9M params, 50-step chunk
# ---------------------------------------------------------------------------
_reg(EvalConfig(
    policy_name="act",
    benchmark="robotwin:beat_block_hammer",
    readiness=Readiness.READY,
    checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/act_avada11_beat_block_hammer/beat_block_hammer.click_bell/demo_clean-100/100",
    checkpoint_note="ACT (DETR-CVAE, 83.9M, resnet18) finetuned on RoboTwin 2.0 beat_block_hammer (Avada11, demo_clean-100, seed 100, 6000 epochs). Single-task per-seed ckpt; dataset_stats.pkl in same dir.",
    server_script="RoboTwin/policy/ACT/policy_server.py",
    server_args={
        "--checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/act_avada11_beat_block_hammer/beat_block_hammer.click_bell/demo_clean-100/100",
        "--task": "beat_block_hammer",
        "--action_dim": "14",
        "--chunk_size": "50",
    },
    arm_controller="joint_pos",
    action_dim=14,
    action_range=(-3.15, 3.15),
    known_issues=[
        "Own-venv pattern: RoboTwin/policy/ACT/.venv (torch 2.4.1+cu121, opencv-headless).",
        "Upstream loader hardcodes policy_last.ckpt; server symlinks first policy_*.ckpt automatically.",
        "Avada11 layout is <pretrained>.<eval>/<setting>/<seed>/policy_sim-<train_task>-...ckpt; pick the ckpt whose train_task matches the eval task.",
    ],
    expected_success_rate=0.40,
    expected_source="RoboTwin 2.0 paper ACT Easy avg ~40%",
))


# ---------------------------------------------------------------------------
#  DP × RoboTwin — Avada11/AgileX same-task Diffusion Policy ckpts
#  reference: Avada11/RoboTwin-Model-AgileX-DP (HF), DDPM 100-step, resnet18
# ---------------------------------------------------------------------------
_reg(EvalConfig(
    policy_name="dp",
    benchmark="robotwin:beat_block_hammer",
    readiness=Readiness.READY,
    checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/dp_avada11_beat_block_hammer/500.ckpt",
    checkpoint_note="Diffusion Policy (resnet18 + 1D UNet, ~260M) finetuned on RoboTwin 2.0 beat_block_hammer (Avada11, demo_clean-100, seed 100, epoch 500). Hydra+dill ckpt, n_obs_steps=3, n_action_steps=6.",
    server_script="RoboTwin/policy/DP/policy_server.py",
    server_args={
        "--checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/dp_avada11_beat_block_hammer/500.ckpt",
        "--task": "beat_block_hammer",
        "--action_dim": "14",
        "--n_obs_steps": "3",
        "--n_action_steps": "6",
    },
    arm_controller="joint_pos",
    action_dim=14,
    action_range=(-3.15, 3.15),
    known_issues=[
        "Own-venv pattern: RoboTwin/policy/DP/.venv (torch 2.4.1+cu121, hydra==1.2.0, dill, diffusers).",
        "Stateful: server keeps a deque of n_obs_steps+1 obs; reset() called on episode boundary.",
        "Avada11 same-task layout: <task>-demo_clean-100/<seed>/<epoch>.ckpt (no cross-task subdir).",
    ],
    expected_success_rate=0.40,
    expected_source="RoboTwin 2.0 paper Diffusion Policy Easy avg ~40%",
))


# ---------------------------------------------------------------------------
#  Pi0.5 × RoboTwin — Hoshipu/30k single-ckpt + Crelf/50-task
# ---------------------------------------------------------------------------
# Pi0.5 × RoboTwin (Hoshipu pi05-robotwin2-clean-30k) — openpi JAX/orbax format
# Layout: assets/{physical-intelligence/libero, robotwin2_clean_ft/norm_stats.json},
#         params/, train_state/, _CHECKPOINT_METADATA. Top-level openpi orbax.
# Cannot load via lerobot/policy_server.py. Needs openpi route + a pi05_robotwin
# train_config (upstream RoboTwin/policy/pi05/src has only pi0_*_aloha_robotwin_*
# configs, no pi05). Upstream wrapper at RoboTwin/policy/pi05/pi_model.py uses
# openpi.policies.policy_config.create_trained_policy with robotwin_repo_id=
# "robotwin2_clean_ft" — would need its own venv (RoboTwin/policy/pi05/.venv
# does not exist). Keeping registered for tracking; readiness blocked until
# the openpi route + pi05_robotwin config + venv are wired.
_reg(EvalConfig(
    policy_name="pi0.5",
    benchmark="robotwin:beat_block_hammer",
    readiness=Readiness.NEEDS_FINETUNE,
    checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/pi05-robotwin2-clean-30k",
    checkpoint_note=(
        "Pi0.5 RoboTwin finetune (Hoshipu, 30k). openpi JAX/orbax — needs openpi "
        "policy_server + pi05_robotwin train_config + RoboTwin/policy/pi05/.venv."
    ),
    server_script="openpi/scripts/policy_server.py",
    server_args={"--checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/pi05-robotwin2-clean-30k"},
    arm_controller="joint_pos",
    action_dim=14,
    action_range=(-3.15, 3.15),
    expected_success_rate=0.50,
    expected_source="RoboTwin 2.0 leaderboard Pi0.5 Easy avg 50%",
))

# Pi0.5 × RoboTwin 50-task democlean (Crelf) — openpi-PyTorch (train_pytorch.py)
# Layout: 35000/{metadata.pt, model.safetensors}, README.md only. metadata.pt
# config: name="C3I_pi05_50tasks_train_config_clean", project_name="openpi",
# action_dim=32, action_horizon=50, pi05=True. PyTorch openpi flavour, distinct
# from JAX flavour. Same blocker as Hoshipu — needs openpi-pytorch loader path.
_reg(EvalConfig(
    policy_name="pi0.5",
    benchmark="robotwin:click_bell",
    readiness=Readiness.NEEDS_FINETUNE,
    checkpoint=f"{_ROBOTWIN_CKPT_ROOT}/C3I_pi05_Robotwin_50tasks_model_democlean",
    checkpoint_note=(
        "Pi0.5 RoboTwin 50-task democlean (Crelf, 35k). openpi-PyTorch — needs "
        "openpi train_pytorch ckpt loader + 32D-action wiring (RoboTwin client emits 14D)."
    ),
    server_script="openpi/scripts/policy_server.py",
    server_args={"--checkpoint": f"{_ROBOTWIN_CKPT_ROOT}/C3I_pi05_Robotwin_50tasks_model_democlean"},
    arm_controller="joint_pos",
    action_dim=14,
    action_range=(-3.15, 3.15),
    expected_success_rate=0.50,
    expected_source="RoboTwin 2.0 leaderboard Pi0.5 Easy avg 50%",
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
