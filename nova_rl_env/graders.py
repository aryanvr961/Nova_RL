# =============================================================================
# File: nova_rl_env/graders.py
# Owner: Aadyaa
# Role: Deterministic scoring logic
# =============================================================================
#
# Purpose
# - This file will implement the deterministic graders for the required tasks.
# - It is responsible for producing final scores in the correct range.
#
# What this file should contain later
# - Shared score helpers.
# - Task-specific grading rules if needed.
# - Final score clamping to 0.0 to 1.0.
#
# Design expectations
# - The same task outcome should always produce the same score.
# - Scoring should be easy to explain and justify.
# - Graders should reflect real task quality, not arbitrary heuristics.
#
# Constraints
# - No randomness in grading.
# - No LLM calls in grading.
# - Keep graders independent from deployment logic.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .models import Action
else:
    Action = Any


def _clamp_score(value: float) -> float:
    return max(0.0, min(1.0, value))


def _metric(state: Mapping[str, Any], name: str) -> float:
    metrics = state.get("metrics", {})
    if not isinstance(metrics, Mapping):
        return 0.0
    return _clamp_score(float(metrics.get(name, 0.0)))


def _action_value(action: Action, name: str, default: Any) -> Any:
    if isinstance(action, Mapping):
        return action.get(name, default)
    return getattr(action, name, default)


def _latency_penalty(state: Mapping[str, Any]) -> float:
    step_index = max(0, int(state.get("step_index", 0)))
    batch = state.get("batch", {})
    max_steps = 8
    if isinstance(batch, Mapping):
        max_steps = int(batch.get("max_steps", max_steps))
    return _clamp_score(step_index / max_steps)


def _quarantine_penalty(action: Action, quarantine_precision: float) -> float:
    decision = _action_value(action, "decision", "")
    threshold = float(_action_value(action, "threshold", 0.5))
    if decision == "quarantine":
        return _clamp_score((1.0 - quarantine_precision) * max(threshold, 0.0))
    return _clamp_score(1.0 - quarantine_precision)


def _grade_easy(state: Mapping[str, Any]) -> float:
    return _metric(state, "promotion_precision")


def _grade_medium(state: Mapping[str, Any], action: Action) -> float:
    promotion_rate = _metric(state, "promotion_precision")
    quarantine_precision = _metric(state, "quarantine_precision")
    quarantine_penalty = _quarantine_penalty(action, quarantine_precision)
    return _clamp_score(0.75 * promotion_rate - 0.25 * quarantine_penalty)


def _grade_hard(state: Mapping[str, Any], action: Action) -> float:
    promotion_rate = _metric(state, "promotion_precision")
    quarantine_precision = _metric(state, "quarantine_precision")
    quarantine_penalty = _quarantine_penalty(action, quarantine_precision)
    latency_penalty = _latency_penalty(state)
    return _clamp_score(
        0.70 * promotion_rate
        - 0.20 * quarantine_penalty
        - 0.10 * latency_penalty
    )


def grade(task_id: str, state: dict, action: Action) -> float:
    """Return a deterministic task score in the required 0.0 to 1.0 range."""

    if task_id == "easy":
        return _grade_easy(state)
    if task_id == "medium":
        return _grade_medium(state, action)
    if task_id == "hard":
        return _grade_hard(state, action)
    raise ValueError(f"Unknown task_id: {task_id}")
