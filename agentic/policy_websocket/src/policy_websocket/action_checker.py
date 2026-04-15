"""ActionSanityChecker — detect and report action space mismatches early.

Catches the three most common integration pitfalls:
1. Dimension mismatch (e.g. 7D Cartesian policy → 9D joint-space env)
2. Scale mismatch (e.g. actions ~0.01 when env expects [-1,1])
3. Unbounded actions (e.g. raw model output >1.0 going into clipped env)

Usage in eval scripts:
    checker = ActionSanityChecker(env_action_dim=8, env_action_low=-1, env_action_high=1)
    for t in range(max_steps):
        action = policy.infer(obs)["actions"]
        action = checker.check(action, t)  # warns + auto-fixes
        env.step(action)
    checker.report()  # print summary at episode end
"""

import logging
from typing import Union

import numpy as np

logger = logging.getLogger(__name__)

_WARN_THRESHOLD = 5  # warn after this many steps of collecting stats
_SCALE_TOO_SMALL = 0.05  # if 90th-percentile |action| < this, likely wrong norm
_SCALE_TOO_LARGE = 2.0   # if 10th-percentile |action| > this, likely unbounded


class ActionSanityChecker:
    """Lightweight action validator that detects mismatches early."""

    def __init__(
        self,
        env_action_dim: int,
        env_action_low: Union[float, np.ndarray] = -1.0,
        env_action_high: Union[float, np.ndarray] = 1.0,
        auto_clip: bool = True,
        policy_name: str = "",
        env_name: str = "",
    ):
        self.env_action_dim = env_action_dim
        self.env_low = np.broadcast_to(np.asarray(env_action_low, dtype=np.float64), (env_action_dim,))
        self.env_high = np.broadcast_to(np.asarray(env_action_high, dtype=np.float64), (env_action_dim,))
        self.auto_clip = auto_clip
        self.policy_name = policy_name
        self.env_name = env_name

        # Stats collection
        self._abs_values = []  # collect |action| for scale check
        self._dim_warned = False
        self._scale_warned = False
        self._clip_warned = False
        self._total_steps = 0
        self._clip_count = 0
        self._dim_mismatch_count = 0

    def check(self, action: np.ndarray, _step: int = 0) -> np.ndarray:
        """Validate action and return (possibly fixed) action."""
        action = np.asarray(action, dtype=np.float64)
        self._total_steps += 1

        # --- Check 1: Dimension mismatch ---
        if action.shape[-1] != self.env_action_dim:
            self._dim_mismatch_count += 1
            if not self._dim_warned:
                self._dim_warned = True
                logger.warning(
                    "ACTION DIM MISMATCH: policy outputs %dD but env expects %dD. "
                    "This often means wrong control mode (e.g. Cartesian policy → joint-space env). "
                    "Policy=%s Env=%s",
                    action.shape[-1], self.env_action_dim,
                    self.policy_name, self.env_name,
                )
            # Pad or truncate
            if action.shape[-1] < self.env_action_dim:
                action = np.pad(action, (0, self.env_action_dim - action.shape[-1]))
            else:
                action = action[:self.env_action_dim]

        # --- Collect stats for scale check (skip all-zero init actions) ---
        abs_action = np.abs(action[:min(7, len(action))])
        if np.any(abs_action > 1e-10):
            self._abs_values.append(abs_action)

        # --- Check 2 & 3: Scale and bounds (after collecting enough non-zero data) ---
        if len(self._abs_values) == _WARN_THRESHOLD:
            abs_arr = np.array(self._abs_values)  # (N, dim)
            p90 = np.percentile(abs_arr, 90)
            median = np.median(abs_arr)

            if p90 < _SCALE_TOO_SMALL and not self._scale_warned:
                self._scale_warned = True
                logger.warning(
                    "ACTION SCALE TOO SMALL: 90th-pct |action|=%.4f (median=%.4f). "
                    "Policy may have wrong denormalization (e.g. bridge stats on LIBERO). "
                    "Expected range: [%.1f, %.1f]. Policy=%s Env=%s",
                    p90, median, self.env_low[0], self.env_high[0],
                    self.policy_name, self.env_name,
                )

            max_val = np.max(abs_arr)
            if max_val > _SCALE_TOO_LARGE and not self._clip_warned:
                self._clip_warned = True
                logger.warning(
                    "ACTION SCALE TOO LARGE: max |action|=%.2f. "
                    "Policy output may be unbounded (raw model output without clipping). "
                    "Expected range: [%.1f, %.1f]. Auto-clip=%s. Policy=%s Env=%s",
                    max_val, self.env_low[0], self.env_high[0],
                    self.auto_clip, self.policy_name, self.env_name,
                )

        # Stop collecting after enough data
        if len(self._abs_values) > _WARN_THRESHOLD * 2:
            self._abs_values = self._abs_values[-_WARN_THRESHOLD:]

        # --- Auto-clip to env bounds ---
        if self.auto_clip:
            clipped = np.clip(action, self.env_low[:len(action)], self.env_high[:len(action)])
            if not np.array_equal(clipped, action):
                self._clip_count += 1
            action = clipped

        return action

    def report(self) -> str:
        """Return a summary string. Call at episode end."""
        lines = []
        if self._dim_warned:
            lines.append(f"  ⚠ DIM MISMATCH: {self._dim_mismatch_count}/{self._total_steps} steps had wrong dim")
        if self._scale_warned:
            lines.append(f"  ⚠ SCALE TOO SMALL: actions near zero — check denormalization/unnorm_key")
        if self._clip_warned:
            lines.append(f"  ⚠ SCALE TOO LARGE: {self._clip_count}/{self._total_steps} steps were clipped")
        if not lines:
            return ""
        header = f"  Action Sanity Report ({self.policy_name} → {self.env_name}):"
        return "\n".join([header] + lines)

    @property
    def has_warnings(self) -> bool:
        return self._dim_warned or self._scale_warned or self._clip_warned
