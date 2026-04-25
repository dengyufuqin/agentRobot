#!/usr/bin/env python3
"""Produce eval_protocol.json for a (policy, benchmark, checkpoint) tuple.

Auto-extraction sources (in order of preference, most reliable first):
  1. **Local checkpoint config scan**: preprocessor_config.json (image size),
     dataset_statistics.json / norm_stats.json (unnorm_key), README.md hints.
  2. **Known (policy, benchmark) templates**: hard-coded per-combo fields that
     we've already validated (e.g. OpenVLA×ManiSkill center_crop=False,
     gripper rescale-only). These come from feedback_*.md memories after
     each alignment effort and are the single source of truth.
  3. **Field-level merge**: if the output file already exists, its cited
     fields are preserved; only [MANUAL] slots get overwritten.

`--validate` gates run_benchmark: every field must have a concrete value
AND a non-empty source citation, else exit 3.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

MANUAL = "[MANUAL]"

REQUIRED_FIELDS = [
    "image_resolution",
    "max_episode_steps",
    "camera",
    "control_mode",
    "obs_mode",
    "prompt_format",
    "image_flip_180deg",
    "resize_method",
    "image_resize_before_model",
    "center_crop",
    "gripper_post_processing",
    "wait_steps",
    "unnorm_key",
    # Checkpoint-shape fields (auto-extracted from config.json). Capture these so
    # the client can catch shape mismatches (e.g. pi0fast trained dual-arm 16D
    # vs RoboCasa client sending 8D) before a run starts.
    "state_dim",
    "action_dim",
    "required_obs_keys",
    "obs_wrapper",
    # Seed-space fields. Train range is extracted from ckpt name (e.g. "seed1k"
    # → [0,1000)); eval range is per-benchmark (RoboTwin run_eval_ws.py bases
    # seeds at 100000 × (1+args.seed) — so args.seed=0 gives [100000, 100000+N)).
    # If ranges don't overlap we evaluate fully OOD and scores collapse to 0,
    # which looks like a policy failure but is actually a harness misconfig.
    "train_seed_range",
    "eval_seed_range",
]

# LeRobot policy types that route through an EnvTransition (preprocessor pulls
# `complementary_data["task"]` → tokenized observation.language.tokens). All
# other policies consume a flat observation dict.
_LEROBOT_VLM_TYPES = {"pi0", "pi05", "pi0_fast", "smolvla", "xvla", "wall_x", "groot"}


# =============================================================================
# Known (policy, benchmark) templates — each value is fully cited. Adding a
# new combo here is how the team captures its alignment knowledge.
# =============================================================================
TEMPLATES: dict[tuple[str, str], dict[str, dict]] = {
    # OpenVLA × ManiSkill (RPD paper — Juelg/openvla-7b-finetuned-maniskill)
    ("openvla", "maniskill"): {
        "image_resolution": {"value": 256, "source": "RPD paper §IV.A: 256×256 human camera render"},
        "max_episode_steps": {"value": 300, "source": "RPD paper §IV.A: max_episode_length=300"},
        "camera": {"value": "HumanCameraWrapper — env.render() overwrites observation['sensor_data']['base_camera']",
                   "source": "vlagents/src/vlagents/wrappers.py:22-29"},
        "control_mode": {"value": "pd_ee_delta_pose", "source": "vlagents/src/vlagents/ppo_rgb_rpd.py:371"},
        "obs_mode": {"value": "rgb", "source": "vlagents/src/vlagents/ppo_rgb_rpd.py:370"},
        "prompt_format": {"value": "In: What action should the robot take to {instruction.lower()}?\\nOut:",
                          "source": "openvla/experiments/robot/openvla_utils.get_prompt_builder()"},
        "image_flip_180deg": {"value": False, "source": "ManiSkill human_render is upright; vlagents does NOT flip"},
        "resize_method": {"value": "lanczos", "source": "HF processor default (vlagents sends 256 raw; processor internal resize)"},
        "image_resize_before_model": {"value": False,
                                      "source": "vlagents/policies.py passes 256×256 directly to processor; NO 256→224 resize"},
        "center_crop": {"value": False,
                        "source": "vlagents/policies.py processor() without center_crop; feedback_openvla_center_crop.md — closed 40pp gap"},
        "gripper_post_processing": {"value": "a[-1] = a[-1] * 2 - 1.0  # rescale [0,1]→[-1,1] ONLY; no binarize, no invert",
                                    "source": "vlagents/src/vlagents/evaluator_envs.py:172"},
        "wait_steps": {"value": 0, "source": "vlagents has no warmup; our harness uses 10 for init-pose alignment"},
        "unnorm_key": {"value": "maniskill_human:7.0.0",
                       "source": "Juelg/openvla-7b-finetuned-maniskill/dataset_statistics.json top-level key"},
        "state_dim": {"value": 8, "source": "OpenVLA ManiSkill RPD: eef_pos(3)+axis_angle(3)+gripper(2)"},
        "action_dim": {"value": 7, "source": "OpenVLA action_head outputs 7-DoF action"},
        "required_obs_keys": {"value": ["image", "state"],
                              "source": "openvla_utils.prepare_images_for_vla + robot_state"},
        "obs_wrapper": {"value": "flat_dict",
                        "source": "OpenVLA processor consumes flat dict; no EnvTransition"},
    },

    # OpenVLA × LIBERO (moojink OFT)
    ("openvla", "libero"): {
        "image_resolution": {"value": 224, "source": "openvla_utils.OPENVLA_IMAGE_SIZE=224"},
        "max_episode_steps": {"value": 220, "source": "LIBERO spec: spatial/object/goal=220; libero_10=520"},
        "camera": {"value": "LIBERO agentview (primary) + robot0_eye_in_hand_image (wrist)",
                   "source": "moojink eval: num_images_in_input=2; LIBERO env default"},
        "control_mode": {"value": "OSC_POSE", "source": "LIBERO env default; client arm_controller=cartesian_pose"},
        "obs_mode": {"value": "rgb", "source": "LIBERO env default"},
        "prompt_format": {"value": "In: What action should the robot take to {instruction.lower()}?\\nOut:",
                          "source": "openvla_utils.get_prompt_builder()"},
        "image_flip_180deg": {"value": True,
                              "source": "LIBERO agentview is upside-down vs training; policy_server.remap_obs_to_libero"},
        "resize_method": {"value": "lanczos3", "source": "openvla_utils.resize_image_for_policy method='lanczos3'"},
        "image_resize_before_model": {"value": True, "source": "prepare_images_for_vla resizes 256→224"},
        "center_crop": {"value": True, "source": "moojink OFT training used center-crop augmentation"},
        "gripper_post_processing": {"value": "_normalize_gripper_action(binarize=True) + _invert_gripper_action",
                                    "source": "openvla/experiments/robot/robot_utils.py; LIBERO -1=open,+1=close"},
        "wait_steps": {"value": 10, "source": "LIBERO eval standard (robosuite settle)"},
        "unnorm_key": {"value": "libero_spatial_no_noops",
                       "source": "moojink/openvla-7b-oft-finetuned-libero-{suite}/dataset_statistics.json"},
        "state_dim": {"value": 8, "source": "LIBERO state: eef_pos(3)+axis_angle(3)+gripper(2)"},
        "action_dim": {"value": 7, "source": "OpenVLA action_head 7-DoF"},
        "required_obs_keys": {"value": ["image_primary", "image_wrist", "state"],
                              "source": "moojink OFT num_images_in_input=2 + proprio"},
        "obs_wrapper": {"value": "flat_dict", "source": "OpenVLA processor consumes flat dict"},
    },

    # OpenVLA-OFT × RoboTwin 2.0 (upstream RoboTwin/policy/openvla-oft/deploy_policy.yml)
    # Ground truth: deploy_policy.yml + finetune_aloha.sh in the upstream RoboTwin repo.
    # Note: Haozhan72 community ckpts are missing action_head.pt and FiLM weights despite
    # this template documenting the upstream-intended config — see feedback memory.
    ("openvla", "robotwin"): {
        "image_resolution": {"value": 224, "source": "OpenVLA image_sizes=[224,224]"},
        "max_episode_steps": {"value": 800, "source": "RoboTwin/script/run_eval_ws.py default rollout limit"},
        "camera": {"value": "RoboTwin 3-camera: head_camera + left_camera + right_camera",
                   "source": "RoboTwin/policy/openvla-oft/deploy_policy.py:32-37 encode_obs"},
        "control_mode": {"value": "joint_position", "source": "RoboTwin ALOHA bimanual joint control"},
        "obs_mode": {"value": "rgb", "source": "RoboTwin default"},
        "prompt_format": {"value": "In: What action should the robot take to {instruction.lower()}?\\nOut:",
                          "source": "OpenVLA standard prompt"},
        "image_flip_180deg": {"value": False, "source": "RoboTwin renders upright; encode_obs passes raw RGB"},
        "resize_method": {"value": "lanczos", "source": "OpenVLA default"},
        "image_resize_before_model": {"value": False, "source": "HF processor handles 224 internally"},
        "center_crop": {"value": True,
                        "source": "RoboTwin/policy/openvla-oft/deploy_policy.yml:14 center_crop=true (matches finetune_aloha.sh --image_aug True)"},
        "gripper_post_processing": {"value": "raw 14D joint output (bimanual ALOHA, no gripper sign convention)",
                                    "source": "dataset_statistics.json action has 14 continuous dims"},
        "wait_steps": {"value": 0, "source": "RoboTwin ws client has no warmup"},
        "unnorm_key": {"value": "[MANUAL]",
                       "source": "task-specific; read top-level key from ckpt's dataset_statistics.json"},
        "state_dim": {"value": 14, "source": "RoboTwin bimanual proprio 14D (OpenVLA-OFT proprio_projector)"},
        "action_dim": {"value": 14, "source": "OpenVLA-OFT 14D action head (NUM_ACTIONS_CHUNK=25)"},
        "required_obs_keys": {"value": ["full_image", "left_wrist_image", "right_wrist_image", "state"],
                              "source": "RoboTwin/policy/openvla-oft/deploy_policy.py:32-37 encode_obs"},
        "obs_wrapper": {"value": "flat_dict", "source": "OpenVLA processor consumes flat dict"},
    },

    # Octo × ManiSkill (RPD — Juelg/octo-base-1.5-finetuned-maniskill)
    ("octo", "maniskill"): {
        "image_resolution": {"value": 256, "source": "RPD paper §IV.A: 256×256 render"},
        "max_episode_steps": {"value": 300, "source": "RPD paper §IV.A: max_episode_length=300"},
        "camera": {"value": "HumanCameraWrapper — observation['sensor_data']['base_camera']=env.render()",
                   "source": "vlagents/src/vlagents/wrappers.py:22-29"},
        "control_mode": {"value": "pd_ee_delta_pose", "source": "vlagents/src/vlagents/ppo_rgb_rpd.py:371"},
        "obs_mode": {"value": "rgb", "source": "vlagents/src/vlagents/ppo_rgb_rpd.py:370"},
        "prompt_format": {"value": "<plain text instruction>  # Octo uses T5 language encoding",
                          "source": "vlagents OctoModel passes raw instruction to Octo tokenizer"},
        "image_flip_180deg": {"value": False, "source": "human_camera is upright"},
        "resize_method": {"value": "lanczos", "source": "Octo pipeline; matches OpenVLA/RPD"},
        "image_resize_before_model": {"value": False, "source": "Octo accepts 256 native (primary=256, wrist=128)"},
        "center_crop": {"value": False, "source": "Juelg/RPD-maniskill training used no center-crop"},
        "gripper_post_processing": {"value": "a[-1] = a[-1] * 2 - 1.0", "source": "vlagents/evaluator_envs.py:172"},
        "wait_steps": {"value": 0, "source": "vlagents has no warmup; our harness uses 10"},
        "unnorm_key": {"value": "maniskill_human:7.0.0",
                       "source": "Juelg/octo-base-1.5-finetuned-maniskill/dataset_statistics.json top-level key"},
        "state_dim": {"value": 8, "source": "Octo ManiSkill proprio: eef_pos+axis_angle+gripper"},
        "action_dim": {"value": 7, "source": "Octo diffusion head 7-DoF"},
        "required_obs_keys": {"value": ["image_primary", "image_wrist", "state"],
                              "source": "Octo image_primary (256) + image_wrist (128) + proprio"},
        "obs_wrapper": {"value": "flat_dict", "source": "Octo policy consumes flat dict"},
    },

    # pi0.5 × LIBERO (physical-intelligence, via lerobot/pi05_libero)
    ("pi0.5", "libero"): {
        "image_resolution": {"value": 224, "source": "lerobot pi0.5 preprocessor_config size.height=224"},
        "max_episode_steps": {"value": 220, "source": "LIBERO spec: spatial/object/goal=220; libero_10=520"},
        "camera": {"value": "observation.image (agentview) + observation.image.wrist",
                   "source": "lerobot pi05_libero config — observation-prefixed keys, not DROID keys"},
        "control_mode": {"value": "OSC_POSE", "source": "LIBERO env default; arm_controller=cartesian_pose"},
        "obs_mode": {"value": "rgb", "source": "LIBERO env default"},
        "prompt_format": {"value": "<plain text instruction>  # PaliGemma tokenizer",
                          "source": "lerobot Pi0.5 forward — tokenizer.apply_chat_template on raw instruction"},
        "image_flip_180deg": {"value": True,
                              "source": "LIBERO agentview upside-down vs training; feedback_lerobot_libero_flip.md"},
        "resize_method": {"value": "bilinear", "source": "lerobot image processor default"},
        "image_resize_before_model": {"value": True, "source": "lerobot image_processor resizes 256→224"},
        "center_crop": {"value": False, "source": "lerobot pi0.5 preprocessor_config: no center_crop"},
        "gripper_post_processing": {"value": "8D state (7 joint + 1 gripper); action gripper LIBERO convention (-1=open)",
                                    "source": "feedback_openpi_libero_vs_droid_obs.md"},
        "wait_steps": {"value": 10, "source": "LIBERO eval standard"},
        "unnorm_key": {"value": "physical-intelligence/libero",
                       "source": "lerobot pi05_libero assets/physical-intelligence/libero/norm_stats.json"},
        "state_dim": {"value": 8, "source": "pi0.5 LIBERO: 7 joint + 1 gripper (feedback_openpi_libero_vs_droid_obs)"},
        "action_dim": {"value": 7, "source": "pi0.5 LIBERO action decoder 7-DoF"},
        "required_obs_keys": {"value": ["observation/image", "observation/wrist_image", "observation/state"],
                              "source": "openpi pi05_libero config observation/* keys"},
        "obs_wrapper": {"value": "flat_dict",
                        "source": "openpi server consumes flat dict (not EnvTransition); tokenization handled in server"},
    },

    # pi0.5 × ManiSkill (lerobot/pi05_maniskill — physical-intelligence asset)
    ("pi0.5", "maniskill"): {
        "image_resolution": {"value": 224, "source": "lerobot pi0.5 preprocessor_config size.height=224"},
        "max_episode_steps": {"value": 300, "source": "RPD-style ManiSkill convention (align with Juelg)"},
        "camera": {"value": "HumanCameraWrapper — env.render() → observation.image",
                   "source": "our ManiSkill harness uses HumanCameraWrapper (matches OpenVLA/Octo RPD path)"},
        "control_mode": {"value": "pd_ee_delta_pose", "source": "ManiSkill RPD convention"},
        "obs_mode": {"value": "rgb", "source": "ManiSkill default"},
        "prompt_format": {"value": "<plain text instruction>  # PaliGemma tokenizer",
                          "source": "lerobot Pi0.5 tokenizer.apply_chat_template"},
        "image_flip_180deg": {"value": False, "source": "ManiSkill human_render is upright"},
        "resize_method": {"value": "bilinear", "source": "lerobot image processor default"},
        "image_resize_before_model": {"value": True, "source": "lerobot image_processor resizes to 224"},
        "center_crop": {"value": False, "source": "lerobot pi0.5 preprocessor_config: no center_crop"},
        "gripper_post_processing": {"value": "LeRobot native action; [-1,+1] range, gripper last dim",
                                    "source": "lerobot pi0.5 action decoder output"},
        "wait_steps": {"value": 10, "source": "ManiSkill harness default"},
        "unnorm_key": {"value": "physical-intelligence/maniskill",
                       "source": "lerobot pi05_maniskill assets/physical-intelligence/maniskill/norm_stats.json"},
        "state_dim": {"value": 8, "source": "pi0.5 ManiSkill proprio: eef+quat+gripper (openpi maniskill config)"},
        "action_dim": {"value": 7, "source": "pi0.5 ManiSkill action decoder 7-DoF"},
        "required_obs_keys": {"value": ["observation/image", "observation/wrist_image", "observation/state"],
                              "source": "openpi pi05_maniskill config observation/* keys"},
        "obs_wrapper": {"value": "flat_dict", "source": "openpi server flat dict"},
    },

    # pi0 × ManiSkill (Dexmal/Dexbotic-PI0 finetuned on ManiSkill)
    ("pi0", "maniskill"): {
        "image_resolution": {"value": 224, "source": "dexmal preprocessor_config.json size.height=224"},
        "max_episode_steps": {"value": 300, "source": "ManiSkill RPD convention"},
        "camera": {"value": "HumanCameraWrapper — env.render() → base_camera",
                   "source": "Dexbotic eval uses ManiSkill human camera (matches our RPD path)"},
        "control_mode": {"value": "pd_ee_delta_pose", "source": "ManiSkill RPD convention"},
        "obs_mode": {"value": "rgb", "source": "ManiSkill default"},
        "prompt_format": {"value": "<plain text instruction>  # Dexbotic PI0 uses Gemma tokenizer",
                          "source": "dexmal-pi0-ms config.json llm_config model_type=gemma"},
        "image_flip_180deg": {"value": False, "source": "ManiSkill human_render is upright"},
        "resize_method": {"value": "bicubic", "source": "SiglipImageProcessor resample=3 (bicubic)"},
        "image_resize_before_model": {"value": True, "source": "preprocessor_config do_resize=true to 224"},
        "center_crop": {"value": False, "source": "preprocessor_config: no do_center_crop flag"},
        "gripper_post_processing": {"value": "action chunk final dim; binarize per Dexbotic playground/maniskill2_pi0.py",
                                    "source": "Dexmal/Dexbotic-PI0 maniskill2_pi0.py benchmark config"},
        "wait_steps": {"value": 10, "source": "ManiSkill harness default"},
        "unnorm_key": {"value": "maniskill2", "source": "dexmal-pi0-ms norm_stats.json"},
        "state_dim": {"value": 8, "source": "Dexbotic-PI0 ManiSkill proprio: eef+quat+gripper"},
        "action_dim": {"value": 7, "source": "Dexbotic pi0 action decoder 7-DoF"},
        "required_obs_keys": {"value": ["image", "wrist_image", "state"],
                              "source": "Dexbotic maniskill2_pi0.py obs format"},
        "obs_wrapper": {"value": "flat_dict", "source": "Dexbotic/openpi server flat dict"},
    },

    # pi0 × LIBERO (lerobot/pi0_libero finetuned — uses openpi path)
    ("pi0", "libero"): {
        "image_resolution": {"value": 224, "source": "lerobot pi0 preprocessor_config size=224"},
        "max_episode_steps": {"value": 220, "source": "LIBERO spec"},
        "camera": {"value": "observation.image (agentview) + observation.image.wrist",
                   "source": "lerobot pi0_libero config — observation-prefixed keys"},
        "control_mode": {"value": "OSC_POSE", "source": "LIBERO default; arm_controller=cartesian_pose"},
        "obs_mode": {"value": "rgb", "source": "LIBERO default"},
        "prompt_format": {"value": "<plain text instruction>  # PaliGemma tokenizer",
                          "source": "lerobot Pi0 forward path"},
        "image_flip_180deg": {"value": True,
                              "source": "feedback_lerobot_libero_flip.md — pi0 needs flip like pi0.5"},
        "resize_method": {"value": "bilinear", "source": "lerobot image processor default"},
        "image_resize_before_model": {"value": True, "source": "lerobot resizes inside processor"},
        "center_crop": {"value": False, "source": "lerobot pi0 preprocessor_config: no center_crop"},
        "gripper_post_processing": {"value": "8D state (7 joint + 1 gripper); LIBERO convention -1=open",
                                    "source": "feedback_openpi_libero_vs_droid_obs.md"},
        "wait_steps": {"value": 10, "source": "LIBERO eval standard"},
        "unnorm_key": {"value": "physical-intelligence/libero",
                       "source": "lerobot pi0_libero assets/physical-intelligence/libero/norm_stats.json"},
        "state_dim": {"value": 8, "source": "pi0 LIBERO: 7 joint + 1 gripper"},
        "action_dim": {"value": 7, "source": "pi0 action decoder 7-DoF"},
        "required_obs_keys": {"value": ["observation/image", "observation/wrist_image", "observation/state"],
                              "source": "openpi pi0_libero config observation/* keys"},
        "obs_wrapper": {"value": "flat_dict", "source": "openpi server flat dict"},
    },

    # pi0fast × RoboCasa (lerobot/pi0fast_robocasa)
    ("pi0fast", "robocasa"): {
        "image_resolution": {"value": 224, "source": "lerobot pi0fast preprocessor size=224"},
        "max_episode_steps": {"value": 500,
                              "source": "RoboCasa benchmark spec — atomic tasks 500 steps (RoboCasa paper §3)"},
        "camera": {"value": "observation.images.robot0_robotview + wrist",
                   "source": "RoboCasa env default + lerobot pi0fast_robocasa config"},
        "control_mode": {"value": "OSC_POSE", "source": "RoboCasa default; arm_controller=cartesian_pose"},
        "obs_mode": {"value": "rgb", "source": "RoboCasa default"},
        "prompt_format": {"value": "<plain text instruction>  # pi0-FAST uses PaliGemma+FAST tokenizer",
                          "source": "lerobot Pi0FastPolicy forward — apply_chat_template on raw instruction"},
        "image_flip_180deg": {"value": False, "source": "RoboCasa renders upright"},
        "resize_method": {"value": "bilinear", "source": "lerobot image processor default"},
        "image_resize_before_model": {"value": True, "source": "lerobot resizes to 224"},
        "center_crop": {"value": False, "source": "lerobot pi0fast preprocessor: no center_crop"},
        "gripper_post_processing": {"value": "gripper last dim; FAST token decode to continuous action",
                                    "source": "lerobot pi0fast action decoder"},
        "wait_steps": {"value": 10, "source": "RoboCasa harness default"},
        "unnorm_key": {"value": "physical-intelligence/robocasa",
                       "source": "lerobot pi0fast_robocasa assets/physical-intelligence/robocasa/norm_stats.json"},
        "state_dim": {"value": 16, "source": "pi0fast RoboCasa trained dual-arm 16D state (pi0fast paper)"},
        "action_dim": {"value": 7, "source": "pi0fast action decoder (FAST tokens → 7D chunk)"},
        "required_obs_keys": {"value": ["observation/image", "observation/wrist_image", "observation/state"],
                              "source": "openpi pi0fast_robocasa config observation/* keys"},
        "obs_wrapper": {"value": "flat_dict", "source": "openpi server flat dict"},
    },

    # smolvla × LIBERO (lerobot/smolvla_base finetuned on aopoli-lv-libero)
    ("smolvla", "libero"): {
        "image_resolution": {"value": 512, "source": "lerobot smolvla preprocessor size=512 (SmolVLM)"},
        "max_episode_steps": {"value": 220, "source": "LIBERO spec"},
        "camera": {"value": "observation.image (agentview) + observation.image.wrist",
                   "source": "lerobot smolvla config — observation-prefixed keys"},
        "control_mode": {"value": "OSC_POSE", "source": "LIBERO default"},
        "obs_mode": {"value": "rgb", "source": "LIBERO default"},
        "prompt_format": {"value": "<plain text instruction>  # SmolVLM tokenizer",
                          "source": "lerobot smolvla tokenizer.apply_chat_template"},
        "image_flip_180deg": {"value": True, "source": "LIBERO agentview is upside-down (same as pi0/pi0.5)"},
        "resize_method": {"value": "bilinear", "source": "lerobot default"},
        "image_resize_before_model": {"value": True, "source": "lerobot resizes to 512"},
        "center_crop": {"value": False, "source": "lerobot smolvla preprocessor: no center_crop"},
        "gripper_post_processing": {"value": "8D state; LIBERO convention -1=open",
                                    "source": "lerobot smolvla LIBERO finetune config"},
        "wait_steps": {"value": 10, "source": "LIBERO eval standard"},
        "unnorm_key": {"value": "aopoli-lv-libero_combined_no_noops",
                       "source": "smolvla-libero README datasets field"},
    },

    # rdt × ManiSkill (Juelg/rdt-1b-finetuned-maniskill equivalent — local rdt-maniskill dir)
    ("rdt", "maniskill"): {
        "image_resolution": {"value": 384, "source": "RDT-1B uses SigLIP-384"},
        "max_episode_steps": {"value": 300, "source": "ManiSkill RPD convention"},
        "camera": {"value": "HumanCameraWrapper — env.render()",
                   "source": "RDT ManiSkill eval matches RPD path"},
        "control_mode": {"value": "pd_ee_delta_pose", "source": "ManiSkill RPD convention"},
        "obs_mode": {"value": "rgb", "source": "ManiSkill default"},
        "prompt_format": {"value": "<plain text instruction>  # RDT uses T5 lang embedding (precomputed in lang_embeds/)",
                          "source": "rdt-maniskill/lang_embeds directory + RDT-1B paper"},
        "image_flip_180deg": {"value": False, "source": "ManiSkill human_render is upright"},
        "resize_method": {"value": "bicubic", "source": "SigLIP default resample=bicubic"},
        "image_resize_before_model": {"value": True, "source": "SigLIP-384 processor resizes to 384"},
        "center_crop": {"value": False, "source": "RDT finetune training used no center-crop"},
        "gripper_post_processing": {"value": "7-DoF + 1 gripper; diffusion policy output",
                                    "source": "RDT-1B action head (64-step diffusion chunk)"},
        "wait_steps": {"value": 10, "source": "ManiSkill harness default"},
        "unnorm_key": {"value": "maniskill", "source": "rdt-maniskill/diffusion_policy norm stats key"},
        "state_dim": {"value": 8, "source": "RDT ManiSkill: eef(3)+axis_angle(3)+gripper(2)"},
        "action_dim": {"value": 7, "source": "RDT-1B action decoder outputs 7-DoF chunk"},
        "required_obs_keys": {"value": ["image_primary", "image_wrist", "state", "lang_embed"],
                              "source": "RDT-1B architecture: img + proprio + precomputed T5 lang"},
        "obs_wrapper": {"value": "flat_dict", "source": "RDT server consumes flat dict"},
    },
}


# =============================================================================
# Per-benchmark eval-seed base extracted from upstream eval scripts. Pair with
# --num-trials at validate time to compute the full [start, end) span.
# =============================================================================
EVAL_SEED_BASES: dict[str, dict[str, Any]] = {
    # RoboTwin 2.0 run_eval_ws.py:180 — now_seed = 100000 * (1 + args.seed).
    # With default --seed 0 this means episode i uses seed 100000+i, which
    # is well outside any "seed1k" (0..999) training distribution.
    "robotwin": {
        "base": 100000,
        "source": "RoboTwin/script/run_eval_ws.py:180 `now_seed = 100000 * (1 + args.seed)` (default --seed 0 → base 100000)",
    },
    # LIBERO: SuiteSeed starts at 0 and increments per init; benchmark task
    # seeds are drawn from [0, ∞) in task-suite order. Training uses same.
    "libero": {
        "base": 0,
        "source": "LIBERO benchmark uses task-suite init seeds starting at 0 (same distribution as training)",
    },
    # ManiSkill: reset() seeds come from env.reset(seed=...) — our harness
    # uses default episode-index seeding (0..num_trials-1).
    "maniskill": {
        "base": 0,
        "source": "ManiSkill harness uses episode-index seeding starting at 0 (aligned with training seeds)",
    },
    # RoboCasa: robosuite seed per episode from episode_id; training uses the
    # same seeding path.
    "robocasa": {
        "base": 0,
        "source": "RoboCasa/robosuite episode-index seeding starting at 0 (aligned with training seeds)",
    },
}


def _parse_train_seed_range_from_name(name: str) -> dict | None:
    """Infer train seed range from ckpt directory name.

    Recognises patterns like:
      - "seed1k"   → [0, 1000)
      - "seed10k"  → [0, 10000)
      - "seeds_0_999" → [0, 1000)
      - "seeds0-4" → [0, 5)

    Returns None if no pattern matches (field stays [MANUAL] for manual fill).
    """
    m = re.search(r"seed(\d+)k", name, flags=re.IGNORECASE)
    if m:
        k = int(m.group(1))
        return {"start": 0, "end": k * 1000}

    m = re.search(r"seeds?[_-](\d+)[_-](\d+)", name, flags=re.IGNORECASE)
    if m:
        lo, hi = int(m.group(1)), int(m.group(2))
        return {"start": lo, "end": hi + 1}

    return None


def _read_json(p: Path) -> dict | None:
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _seed_from_checkpoint(checkpoint: str) -> dict[str, dict]:
    """Scan local checkpoint dir for config-derivable fields."""
    fields: dict[str, dict[str, Any]] = {}
    ckpt_dir = Path(checkpoint)
    if not ckpt_dir.is_dir():
        return fields

    ds_stats = _read_json(ckpt_dir / "dataset_statistics.json")
    if ds_stats and list(ds_stats.keys()):
        fields["unnorm_key"] = {
            "value": list(ds_stats.keys())[0],
            "source": f"{ckpt_dir.name}/dataset_statistics.json top-level key",
        }

    norm_stats = _read_json(ckpt_dir / "norm_stats.json")
    if norm_stats and list(norm_stats.keys()) and "unnorm_key" not in fields:
        fields["unnorm_key"] = {
            "value": list(norm_stats.keys())[0],
            "source": f"{ckpt_dir.name}/norm_stats.json top-level key",
        }

    preproc = _read_json(ckpt_dir / "preprocessor_config.json")
    if preproc:
        size = preproc.get("size")
        if isinstance(size, dict) and "height" in size:
            fields["image_resolution"] = {
                "value": int(size["height"]),
                "source": f"{ckpt_dir.name}/preprocessor_config.json size.height",
            }
        if preproc.get("do_center_crop") is False:
            fields["center_crop"] = {
                "value": False,
                "source": f"{ckpt_dir.name}/preprocessor_config.json do_center_crop=false",
            }
        elif preproc.get("do_center_crop") is True:
            fields["center_crop"] = {
                "value": True,
                "source": f"{ckpt_dir.name}/preprocessor_config.json do_center_crop=true",
            }

    # config.json: state_dim / action_dim / required_obs_keys / obs_wrapper.
    # LeRobot format: top-level `input_features` / `output_features` / `type`.
    cfg = _read_json(ckpt_dir / "config.json")
    if cfg:
        in_feats = cfg.get("input_features") or {}
        out_feats = cfg.get("output_features") or {}
        model_type = cfg.get("type")

        state_feat = in_feats.get("observation.state")
        if isinstance(state_feat, dict):
            shape = state_feat.get("shape")
            if isinstance(shape, (list, tuple)) and shape:
                fields["state_dim"] = {
                    "value": int(shape[0]),
                    "source": f"{ckpt_dir.name}/config.json input_features['observation.state'].shape[0]",
                }

        action_feat = out_feats.get("action")
        if isinstance(action_feat, dict):
            shape = action_feat.get("shape")
            if isinstance(shape, (list, tuple)) and shape:
                fields["action_dim"] = {
                    "value": int(shape[0]),
                    "source": f"{ckpt_dir.name}/config.json output_features['action'].shape[0]",
                }

        if in_feats:
            keys = sorted(in_feats.keys())
            fields["required_obs_keys"] = {
                "value": keys,
                "source": f"{ckpt_dir.name}/config.json input_features.keys()",
            }

        if model_type:
            wrapper = "env_transition" if model_type in _LEROBOT_VLM_TYPES else "flat_dict"
            fields["obs_wrapper"] = {
                "value": wrapper,
                "source": (f"config.json type={model_type} — {'lerobot VLM preprocessor needs '
                           'complementary_data[\"task\"] for tokenization' if wrapper == 'env_transition'
                           else 'non-lerobot policy; server consumes flat obs dict'}"),
            }

    # Train seed range — parsed from ckpt dir name (e.g. "seed1k" → [0, 1000)).
    # If no marker is found we record "unknown" so the overlap gate skips instead
    # of falsely blocking full-space finetunes (RDT, pi0, etc.).
    rng = _parse_train_seed_range_from_name(ckpt_dir.name)
    if rng:
        fields["train_seed_range"] = {
            "value": rng,
            "source": f"parsed from ckpt dir name '{ckpt_dir.name}' (seed<N>k → [0, N*1000))",
        }
    else:
        fields["train_seed_range"] = {
            "value": "unknown",
            "source": f"ckpt dir '{ckpt_dir.name}' has no seed<N>k / seeds_<lo>_<hi> marker; "
                       "assumed full seed space — overlap gate will skip",
        }

    return fields


def _seed_eval_range(benchmark: str, num_trials: int) -> dict[str, dict] | None:
    """Build eval_seed_range field from per-benchmark base + trial count."""
    base_info = EVAL_SEED_BASES.get(benchmark)
    if not base_info:
        return None
    start = int(base_info["base"])
    return {
        "eval_seed_range": {
            "value": {"start": start, "end": start + int(num_trials)},
            "source": f"{base_info['source']}; num_trials={num_trials}",
        }
    }


# Per-benchmark CLI flag that shifts the eval starting seed into an arbitrary
# range. Set up by patching the upstream eval script (see RoboTwin/script/
# run_eval_ws.py --seed_base). Used to auto-fix train/eval seed mismatches.
SEED_BASE_FLAG: dict[str, str] = {
    "robotwin": "--seed_base",
}


def _maybe_autofix_seed_mismatch(
    benchmark: str, fields: dict[str, dict], num_trials: int
) -> None:
    """If train/eval seed ranges don't overlap AND we know the override flag
    for this benchmark, emit `recommended_client_flags` so run_benchmark can
    inject the right --seed_base to bring eval inside the training range.
    Mutates `fields` in place."""
    train = fields.get("train_seed_range", {}).get("value")
    evalr = fields.get("eval_seed_range", {}).get("value")
    flag_name = SEED_BASE_FLAG.get(benchmark)
    if not (isinstance(train, dict) and isinstance(evalr, dict) and flag_name):
        return
    ts, te = train.get("start"), train.get("end")
    es, ee = evalr.get("start"), evalr.get("end")
    if not (isinstance(ts, int) and isinstance(te, int)
            and isinstance(es, int) and isinstance(ee, int)):
        return
    if te > es and ee > ts:
        return  # already overlapping, nothing to fix

    # Shift: pick train_start as the new eval base (in-distribution episodes
    # 0..num_trials-1 inside the trained range).
    new_base = ts
    fields["recommended_client_flags"] = {
        "value": [flag_name, str(new_base)],
        "source": (
            f"auto-fix: train_seed_range [{ts},{te}) and default eval_seed_range "
            f"[{es},{ee}) are disjoint. Shifting eval base to {new_base} via "
            f"`{flag_name} {new_base}` so the first {num_trials} episodes fall "
            f"inside the trained seed range."
        ),
    }
    # Update eval_seed_range to reflect the patched base so the overlap gate
    # now passes with citation.
    fields["eval_seed_range"] = {
        "value": {"start": new_base, "end": new_base + num_trials},
        "source": fields["eval_seed_range"]["source"]
        + f"; auto-shifted via {flag_name} {new_base}",
    }


def _merge(primary: dict[str, dict], fallback: dict[str, dict]) -> dict[str, dict]:
    """`primary` wins if its value is concrete; else fall back."""
    out: dict[str, dict] = {f: {"value": MANUAL, "source": ""} for f in REQUIRED_FIELDS}
    for name in REQUIRED_FIELDS:
        for src in (primary, fallback):
            f = src.get(name)
            if f and f.get("value") not in (None, MANUAL):
                out[name] = f
                break
    return out


def _validate(fields: dict[str, dict]) -> list[str]:
    errors = []
    for name in REQUIRED_FIELDS:
        f = fields.get(name)
        if f is None:
            errors.append(f"{name}: field missing from JSON")
            continue
        if f.get("value") == MANUAL or f.get("value") is None:
            errors.append(f"{name}: value is {MANUAL} (fill from paper/README/eval-repo)")
            continue
        if not f.get("source"):
            errors.append(f"{name}: source citation empty (required for audit)")

    # Seed-space overlap gate — if both ranges are concrete, reject non-overlap.
    train = fields.get("train_seed_range", {}).get("value")
    evalr = fields.get("eval_seed_range", {}).get("value")
    if isinstance(train, dict) and isinstance(evalr, dict):
        ts, te = train.get("start"), train.get("end")
        es, ee = evalr.get("start"), evalr.get("end")
        if (isinstance(ts, int) and isinstance(te, int)
                and isinstance(es, int) and isinstance(ee, int)):
            if te <= es or ee <= ts:
                errors.append(
                    f"train_seed_range [{ts},{te}) ∩ eval_seed_range [{es},{ee}) is EMPTY — "
                    "eval runs entirely OOD for this ckpt. Fix: either pass --seed to shift "
                    "eval base into training range, or patch the eval script's seed formula "
                    "(e.g. RoboTwin run_eval_ws.py:180)."
                )
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--eval-repo", default=None)
    ap.add_argument("--paper-url", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--num-trials", type=int, default=3,
                    help="Trial count, used to compute eval_seed_range end bound")
    args = ap.parse_args()

    agent_root = os.environ.get("AGENTROBOT_ROOT", "/mnt/vast/home/yd66byne/code/agentRobot")
    bench_short = args.benchmark.split(":")[0] if ":" in args.benchmark else args.benchmark
    if bench_short.startswith("libero_"):
        bench_short = "libero"
    default_out = Path(agent_root) / "agentic/robot_agent/eval_protocols" / f"{args.policy}_{bench_short}.json"
    out_path = Path(args.out) if args.out else default_out

    if args.validate:
        if not out_path.is_file():
            print(f"❌ eval_protocol.json not found: {out_path}", file=sys.stderr)
            print(f"   Run: /skill extract_eval_protocol --policy {args.policy} "
                  f"--benchmark {bench_short} --checkpoint <ckpt>", file=sys.stderr)
            return 2
        protocol = json.loads(out_path.read_text())
        errors = _validate(protocol.get("fields", {}))
        if errors:
            print(f"❌ eval_protocol.json has {len(errors)} unresolved field(s):", file=sys.stderr)
            for e in errors:
                print(f"   {e}", file=sys.stderr)
            return 3
        print(f"✓ eval_protocol.json validated: {out_path}")
        return 0

    # Auto-extraction pipeline: local scan → template → eval-seed base → existing.
    scan_fields = _seed_from_checkpoint(args.checkpoint)
    template_fields = TEMPLATES.get((args.policy, bench_short), {})

    # Attach eval_seed_range from per-benchmark base + --num-trials span.
    eval_range_fields = _seed_eval_range(bench_short, args.num_trials) or {}

    # Preference: scan > template (local configs are ground truth),
    # but template wins over MANUAL-only scan fields. eval_seed_range is
    # strictly derived from benchmark script + trial count, so it goes in
    # as a template-level fallback.
    merged = _merge(scan_fields, {**template_fields, **eval_range_fields})

    # Preserve manually-edited fields in the existing file.
    # Skip auto-computed seed fields so rerunning the skill is idempotent
    # (autofix needs to see the raw eval_seed_range each time, not the
    # already-shifted one written during a previous run).
    _AUTO_COMPUTED = {"train_seed_range", "eval_seed_range", "recommended_client_flags"}
    if out_path.is_file():
        try:
            existing = json.loads(out_path.read_text())
            for name, f in existing.get("fields", {}).items():
                if name in _AUTO_COMPUTED:
                    continue
                if f.get("value") not in (None, MANUAL) and name in REQUIRED_FIELDS:
                    merged[name] = f
        except json.JSONDecodeError:
            pass

    # Auto-fix: if train/eval seed ranges are disjoint AND we know how to
    # override the eval base (e.g. RoboTwin --seed_base), emit recommended
    # client flags and patch the eval_seed_range to reflect the shifted base
    # so the overlap gate passes downstream.
    _maybe_autofix_seed_mismatch(bench_short, merged, args.num_trials)

    protocol = {
        "policy": args.policy,
        "benchmark": bench_short,
        "checkpoint": args.checkpoint,
        "eval_repo": args.eval_repo,
        "paper_url": args.paper_url,
        "fields": merged,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"Wrote {out_path}")

    errors = _validate(protocol["fields"])
    if errors:
        print(f"\n[{len(errors)} field(s) still need source citations]")
        for e in errors[:8]:
            print(f"  - {e}")
        print("Fill them from paper/README/eval-repo, then re-run with --validate.")
        return 1
    print("✓ All fields cited — ready for --validate gate.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
