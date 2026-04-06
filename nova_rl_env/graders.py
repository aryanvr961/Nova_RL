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

from typing import Any, Mapping


TASK_WEIGHTS = {
    "easy": {
        "fix_accuracy": 0.65,
        "promotion_precision": 0.10,
        "quarantine_precision": 0.25,
    },
    "medium": {
        "fix_accuracy": 0.55,
        "promotion_precision": 0.25,
        "quarantine_precision": 0.20,
    },
    "hard": {
        "fix_accuracy": 0.45,
        "promotion_precision": 0.25,
        "quarantine_precision": 0.30,
    },
}

DECISION_BONUS = {
    "fix": 0.04,
    "quarantine": 0.03,
    "promote": 0.03,
    "finalize": 0.02,
    "noop": -0.06,
}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def grade(*, task_id: str, state: Mapping[str, Any], action: Any) -> float:
    """Compute a deterministic final score from environment metrics."""

    metrics = state.get("metrics", {})
    if not isinstance(metrics, Mapping):
        metrics = {}

    weights = TASK_WEIGHTS.get(task_id, TASK_WEIGHTS["easy"])
    score = sum(
        float(metrics.get(metric_name, 0.0)) * float(weight)
        for metric_name, weight in weights.items()
    )

    decision = getattr(action, "decision", None)
    if isinstance(decision, str):
        score += DECISION_BONUS.get(decision, 0.0)

    step_index = int(state.get("step_index", 0))
    if step_index > 1:
        score -= min(0.08, (step_index - 1) * 0.01)

    batch = state.get("batch", {})
    if isinstance(batch, Mapping):
        anomaly_counts = batch.get("anomaly_counts", {})
        if isinstance(anomaly_counts, Mapping):
            anomaly_total = sum(int(value) for value in anomaly_counts.values())
            if anomaly_total == 0:
                score -= 0.05

    return _clamp(score)
