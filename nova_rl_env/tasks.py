# =============================================================================
# File: nova_rl_env/tasks.py
# Owner: Aadyaa
# Role: Task definitions and difficulty structure
# =============================================================================
#
# Purpose
# - This file will define the easy, medium, and hard tasks required by the hackathon.
# - It should describe how task difficulty changes across the benchmark.
#
# What this file should contain later
# - Task names.
# - Task objectives.
# - Difficulty-specific settings such as anomaly rate or fault complexity.
# - Any static metadata needed by the environment or inference loop.
#
# Design expectations
# - Each task must represent a concrete real-world objective.
# - Difficulty should increase meaningfully, not randomly.
# - Tasks should feel like one coherent benchmark family.
#
# Constraints
# - Keep the MVP compact.
# - Do not add unnecessary task sprawl before the required three tasks are solid.

from __future__ import annotations

from typing import Any


TASKS: dict[str, dict[str, Any]] = {
    "easy": {
        "objective": "Handle null values and exact duplicate rows safely.",
        "anomaly_rate": 0.05,
        "batch_size": 100,
        "max_steps": 8,
        "allowed_anomalies": ["null", "duplicate"],
        "target_strategy": "Prioritize fixing obvious data quality issues with minimal risk.",
    },
    "medium": {
        "objective": "Handle type mismatches and malformed date values.",
        "anomaly_rate": 0.12,
        "batch_size": 120,
        "max_steps": 8,
        "allowed_anomalies": ["null", "duplicate", "type_mismatch", "malformed_date"],
        "target_strategy": "Balance corrective fixes with safe promotion of valid rows.",
    },
    "hard": {
        "objective": "Handle schema drift and correlated multi-column issues.",
        "anomaly_rate": 0.18,
        "batch_size": 140,
        "max_steps": 8,
        "allowed_anomalies": [
            "null",
            "duplicate",
            "type_mismatch",
            "malformed_date",
            "schema_drift",
        ],
        "target_strategy": "Handle mixed anomalies without over-quarantining recoverable rows.",
    },
}


def get_task_config(task_id: str) -> dict[str, Any]:
    """Return a detached task config for the requested benchmark level."""

    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return dict(TASKS[task_id])
