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

from typing import Any, Dict


TASK_ORDER = ["easy", "medium", "hard"]

TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "name": "Easy ETL Nulls and Duplicates",
        "level": "easy",
        "objective": "Fix null fields and obvious duplicate records before promotion.",
        "batch_size": 100,
        "anomaly_rate": 0.10,
        "anomaly_types": ["null", "duplicate"],
        "fault_complexity": "single-row obvious faults",
        "schema_drift_prob": 0.0,
        "max_steps": 8,
        "scoring": {
            "method": "promotion_rate",
            "promotion_weight": 1.0,
            "quarantine_penalty_weight": 0.0,
            "latency_penalty_weight": 0.0,
        },
    },
    "medium": {
        "name": "Medium ETL Type and Date Repair",
        "level": "medium",
        "objective": "Fix type mismatches and malformed date values while avoiding over-quarantine.",
        "batch_size": 100,
        "anomaly_rate": 0.20,
        "anomaly_types": [
            "null",
            "duplicate",
            "type_mismatch",
            "malformed_date",
        ],
        "fault_complexity": "mixed row-level data quality faults",
        "schema_drift_prob": 0.0,
        "max_steps": 8,
        "scoring": {
            "method": "weighted_promotion_with_quarantine_penalty",
            "promotion_weight": 0.75,
            "quarantine_penalty_weight": 0.25,
            "latency_penalty_weight": 0.0,
        },
    },
    "hard": {
        "name": "Hard ETL Schema Drift and Correlated Faults",
        "level": "hard",
        "objective": "Handle schema drift and correlated multi-column faults without unsafe promotion.",
        "batch_size": 100,
        "anomaly_rate": 0.30,
        "anomaly_types": [
            "null",
            "duplicate",
            "type_mismatch",
            "malformed_date",
            "schema_drift",
        ],
        "fault_complexity": "schema drift plus correlated multi-column faults",
        "schema_drift_prob": 0.10,
        "max_steps": 8,
        "scoring": {
            "method": "full_r_system",
            "alpha": 0.70,
            "beta": 0.20,
            "gamma": 0.10,
        },
    },
}


def get_task_config(task_id: str) -> Dict[str, Any]:
    """Return a copy of the task configuration for the requested difficulty."""

    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")
    return dict(TASKS[task_id])
