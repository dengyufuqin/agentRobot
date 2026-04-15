"""EvalPreflightChecker — smart evaluation gate.

Runs a lightweight pre-flight sequence before committing to a full evaluation:

    1. Registry Check   — Is this model×benchmark combo known? Ready? Needs finetune?
    2. Server Health     — Is the policy server reachable and returning actions?
    3. Smoke Test        — Run 1 episode with ActionSanityChecker and verify:
                           a) action dimensions match the environment
                           b) action scale is reasonable (not ~0 or unbounded)
                           c) no crashes or timeouts
    4. Go/No-Go Report   — Clear verdict with specific fix suggestions if needed

Usage:
    from policy_websocket.eval_preflight import EvalPreflightChecker

    preflight = EvalPreflightChecker(
        policy_name="spatialvla",
        benchmark="maniskill",
        env_action_dim=7,
        env_action_low=-1.0,
        env_action_high=1.0,
    )
    verdict = preflight.check_registry()
    # verdict.ok == False → don't even start the server
    # verdict.ok == True  → proceed to smoke test

    # After running 1 episode with ActionSanityChecker:
    verdict = preflight.evaluate_smoke(checker, episode_success=False, episode_length=40)
    if verdict.should_proceed:
        run_full_evaluation()
    else:
        print(verdict.report)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from policy_websocket.action_checker import ActionSanityChecker
from policy_websocket.eval_registry import (
    EVAL_REGISTRY,
    EvalConfig,
    Readiness,
    lookup,
)

logger = logging.getLogger(__name__)


@dataclass
class PreflightVerdict:
    """Result of a preflight check."""
    ok: bool                          # passed this check?
    should_proceed: bool              # should we run full eval?
    stage: str                        # "registry", "smoke", "overall"
    readiness: Optional[Readiness] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    config: Optional[EvalConfig] = None

    @property
    def report(self) -> str:
        lines = [f"  Preflight [{self.stage}]: {'PASS' if self.ok else 'FAIL'}"]
        if self.readiness:
            lines.append(f"  Readiness: {self.readiness.value}")
        for w in self.warnings:
            lines.append(f"  ⚠ {w}")
        for e in self.errors:
            lines.append(f"  ✘ {e}")
        for s in self.suggestions:
            lines.append(f"  → {s}")
        if self.should_proceed:
            lines.append("  ✓ Proceeding to full evaluation")
        else:
            lines.append("  ✗ Full evaluation NOT recommended")
        return "\n".join(lines)


class EvalPreflightChecker:
    """Smart evaluation gate that prevents wasted GPU hours."""

    def __init__(
        self,
        policy_name: str,
        benchmark: str,
        env_action_dim: int = 7,
        env_action_low: float = -1.0,
        env_action_high: float = 1.0,
        allow_cross_domain: bool = False,
    ):
        self.policy_name = policy_name.lower()
        self.benchmark = benchmark.lower()
        self.env_action_dim = env_action_dim
        self.env_action_low = env_action_low
        self.env_action_high = env_action_high
        self.allow_cross_domain = allow_cross_domain

    def check_registry(self) -> PreflightVerdict:
        """Step 1: Check the model×benchmark compatibility registry."""
        cfg = lookup(self.policy_name, self.benchmark)

        if cfg is None:
            return PreflightVerdict(
                ok=True,
                should_proceed=True,
                stage="registry",
                warnings=["No registry entry for this combo — proceeding with caution"],
                suggestions=["Add an entry to eval_registry.py after this run"],
            )

        verdict = PreflightVerdict(
            ok=True,
            should_proceed=True,
            stage="registry",
            readiness=cfg.readiness,
            config=cfg,
        )

        if cfg.readiness == Readiness.READY:
            if cfg.expected_success_rate is not None:
                verdict.suggestions.append(
                    f"Expected ~{cfg.expected_success_rate:.0%} ({cfg.expected_source})"
                )

        elif cfg.readiness == Readiness.NEEDS_FINETUNE:
            verdict.ok = False
            verdict.should_proceed = False
            verdict.errors.append(
                f"This model needs fine-tuning on {self.benchmark} before evaluation"
            )
            if cfg.checkpoint_note:
                verdict.errors.append(f"Current checkpoint: {cfg.checkpoint_note}")
            if cfg.expected_success_rate is not None:
                verdict.suggestions.append(
                    f"Without finetune, expect ~{cfg.expected_success_rate:.0%}"
                )
            verdict.suggestions.append(
                "Fine-tune on target benchmark data first, or use --allow-cross-domain to run anyway"
            )

        elif cfg.readiness == Readiness.CROSS_DOMAIN:
            if self.allow_cross_domain:
                verdict.warnings.append(
                    "Cross-domain evaluation — expect ~0% (running as generalization baseline)"
                )
            else:
                verdict.ok = False
                verdict.should_proceed = False
                verdict.errors.append(
                    f"Cross-domain: {self.policy_name} was not trained on {self.benchmark}"
                )
                verdict.suggestions.append(
                    "Use --allow-cross-domain to run anyway as a generalization baseline"
                )

        elif cfg.readiness == Readiness.UNSUPPORTED:
            verdict.ok = False
            verdict.should_proceed = False
            verdict.errors.append("This combination is known to be incompatible")

        # Add known issues as warnings
        for issue in cfg.known_issues:
            verdict.warnings.append(issue)

        return verdict

    def evaluate_smoke(
        self,
        checker: ActionSanityChecker,
        episode_success: bool,
        episode_length: int,
        max_steps: int = 500,
    ) -> PreflightVerdict:
        """Step 2: Evaluate smoke test results from ActionSanityChecker.

        Call this after running 1 episode with the checker.
        """
        verdict = PreflightVerdict(
            ok=True,
            should_proceed=True,
            stage="smoke",
        )

        # Check for dimension warnings
        if checker._dim_warned:
            verdict.ok = False
            verdict.should_proceed = False
            verdict.errors.append(
                f"ACTION DIM MISMATCH: policy outputs wrong dimensions "
                f"({checker._dim_mismatch_count}/{checker._total_steps} steps)"
            )
            verdict.suggestions.append(
                "Check control_mode (ManiSkill) or arm_controller (LIBERO/RoboCasa)"
            )

        # Check for scale warnings
        if checker._scale_warned:
            verdict.ok = False
            verdict.should_proceed = False
            verdict.errors.append(
                "ACTION SCALE TOO SMALL: actions near zero — likely wrong denormalization"
            )
            verdict.suggestions.append(
                "Check unnorm_key / action normalization stats in policy server"
            )

        # Check for clipping warnings (too large)
        if checker._clip_warned:
            verdict.warnings.append(
                f"ACTIONS CLIPPED: {checker._clip_count}/{checker._total_steps} steps "
                f"exceeded env bounds — auto-clipped"
            )
            verdict.suggestions.append(
                "Consider adding np.clip() in policy server to avoid silent clipping"
            )

        # Check if episode ended immediately (possible crash/config issue)
        if episode_length < 5 and not episode_success:
            verdict.warnings.append(
                f"Very short episode ({episode_length} steps) — possible env config issue"
            )

        # If episode succeeded in smoke test, definitely proceed
        if episode_success:
            verdict.suggestions.append(
                f"Smoke test succeeded at step {episode_length} — model appears functional"
            )

        return verdict

    def full_verdict(
        self,
        registry_verdict: PreflightVerdict,
        smoke_verdict: Optional[PreflightVerdict] = None,
    ) -> PreflightVerdict:
        """Combine registry + smoke verdicts into final go/no-go."""
        verdict = PreflightVerdict(
            ok=True,
            should_proceed=True,
            stage="overall",
            readiness=registry_verdict.readiness,
            config=registry_verdict.config,
        )

        # Merge all warnings/errors/suggestions
        verdict.warnings.extend(registry_verdict.warnings)
        verdict.errors.extend(registry_verdict.errors)
        verdict.suggestions.extend(registry_verdict.suggestions)

        if smoke_verdict:
            verdict.warnings.extend(smoke_verdict.warnings)
            verdict.errors.extend(smoke_verdict.errors)
            verdict.suggestions.extend(smoke_verdict.suggestions)

        # If either check failed hard, don't proceed
        if not registry_verdict.should_proceed:
            verdict.ok = False
            verdict.should_proceed = False
        if smoke_verdict and not smoke_verdict.should_proceed:
            verdict.ok = False
            verdict.should_proceed = False

        return verdict
