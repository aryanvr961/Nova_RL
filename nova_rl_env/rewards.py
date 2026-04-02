# =============================================================================
# File: nova_rl_env/rewards.py
# Owner: Aryan
# Role: Reward design and shaping
# =============================================================================
#
# Purpose
# - Centralizes reward composition for the environment.
# - Keeps reward shaping deterministic, interpretable, and reusable.
#
# Notes
# - This module is intentionally independent from task-specific grading logic.
# - It provides helpers that `environment.py` can call on every step to ensure
#   the reward signal is dense across the full trajectory.

from typing import Any, Dict

from .models import Reward


def clamp_non_negative(value: float) -> float:
    """Prevents accidental negative gains or penalties where not intended."""

    return max(0.0, value)


def compute_reward_value(
    *,
    correct_fix_gain: float,
    unsafe_promotion_penalty: float,
    over_quarantine_penalty: float,
    step_penalty: float,
) -> float:
    """Combines reward components into a single scalar step reward."""

    return (
        correct_fix_gain
        - unsafe_promotion_penalty
        - over_quarantine_penalty
        - step_penalty
    )


def build_reward(
    *,
    correct_fix_gain: float = 0.0,
    unsafe_promotion_penalty: float = 0.0,
    over_quarantine_penalty: float = 0.0,
    step_penalty: float = 0.0,
    metadata: Dict[str, Any] | None = None,
) -> Reward:
    """Builds the typed reward payload returned by the environment."""

    normalized_gain = clamp_non_negative(correct_fix_gain)
    normalized_unsafe_penalty = clamp_non_negative(unsafe_promotion_penalty)
    normalized_quarantine_penalty = clamp_non_negative(over_quarantine_penalty)
    normalized_step_penalty = clamp_non_negative(step_penalty)

    return Reward(
        value=compute_reward_value(
            correct_fix_gain=normalized_gain,
            unsafe_promotion_penalty=normalized_unsafe_penalty,
            over_quarantine_penalty=normalized_quarantine_penalty,
            step_penalty=normalized_step_penalty,
        ),
        correct_fix_gain=normalized_gain,
        unsafe_promotion_penalty=normalized_unsafe_penalty,
        over_quarantine_penalty=normalized_quarantine_penalty,
        step_penalty=normalized_step_penalty,
        metadata=metadata or {},
    )
