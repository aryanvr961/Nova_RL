# =============================================================================
# File: nova_rl_env/datagen.py
# Owner: Aadyaa
# Role: Synthetic data generation
# =============================================================================
#
# Purpose
# - This file will generate deterministic synthetic ETL batches for each task.
# - It should produce the noisy data conditions that the environment will expose.
#
# What this file should contain later
# - Fixed-seed generation logic.
# - Task-aware anomaly injection.
# - Batch construction for easy, medium, and hard scenarios.
#
# Design expectations
# - Generated batches should resemble realistic data remediation problems.
# - The generator should remain lightweight and reproducible.
# - The output should be easy for the environment and graders to consume.
#
# Constraints
# - Avoid heavy dependencies in MVP.
# - Avoid uncontrolled randomness.

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


ANOMALY_TYPES = [
    "none",
    "null",
    "duplicate",
    "type_mismatch",
    "malformed_date",
    "schema_drift",
]

TASK_ANOMALIES = {
    "easy": ["null", "duplicate"],
    "medium": ["null", "duplicate", "type_mismatch", "malformed_date"],
    "hard": ["null", "duplicate", "type_mismatch", "malformed_date", "schema_drift"],
}


def _build_base_frame(base_count: int, rng: np.random.Generator) -> pd.DataFrame:
    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=base_count,
        freq="h",
    ).astype(str)
    sensor_ids = rng.choice(
        [f"SENSOR_{idx:03d}" for idx in range(1, 21)],
        size=base_count,
        replace=True,
    )
    readings = np.round(rng.normal(loc=50.0, scale=12.5, size=base_count), 2)
    statuses = rng.choice(
        ["OK", "WARN", "FAIL"],
        size=base_count,
        p=[0.75, 0.20, 0.05],
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "sensor_id": sensor_ids,
            "reading": readings.astype(object),
            "status": statuses.astype(object),
        }
    )
    df["value"] = df["reading"]
    df["type_valid"] = True
    df["anomaly_score"] = 0.0
    df["anomaly_type"] = "none"
    return df


def _choose_available_indices(
    available: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if count <= 0 or available.size == 0:
        return np.array([], dtype=int)
    count = min(count, available.size)
    return rng.choice(available, size=count, replace=False)


def _rate_to_count(base_count: int, rate: float) -> int:
    if rate <= 0:
        return 0
    return max(1, int(round(base_count * rate)))


def generate_dirty_data(
    n_rows: int = 100,
    missing_rate: float = 0.10,
    duplicate_rate: float = 0.05,
    type_mismatch_rate: float = 0.05,
    malformed_date_rate: float = 0.05,
    schema_drift_rate: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate deterministic synthetic sensor data with ETL anomalies.

    The resulting DataFrame keeps the total number of rows equal to ``n_rows``
    and includes the required metadata columns for anomaly-aware grading.
    """
    if n_rows <= 0:
        raise ValueError("n_rows must be a positive integer.")
    if not 0 <= missing_rate < 1:
        raise ValueError("missing_rate must be in the range [0, 1).")
    if not 0 <= duplicate_rate < 1:
        raise ValueError("duplicate_rate must be in the range [0, 1).")
    if not 0 <= type_mismatch_rate < 1:
        raise ValueError("type_mismatch_rate must be in the range [0, 1).")
    if not 0 <= malformed_date_rate < 1:
        raise ValueError("malformed_date_rate must be in the range [0, 1).")
    if not 0 <= schema_drift_rate < 1:
        raise ValueError("schema_drift_rate must be in the range [0, 1).")

    rng = np.random.default_rng(random_state)

    duplicate_count = min(int(round(n_rows * duplicate_rate)), max(0, n_rows - 1))
    base_count = n_rows - duplicate_count
    df = _build_base_frame(base_count, rng)

    remaining = np.ones(base_count, dtype=bool)

    null_count = int(round(base_count * missing_rate))
    null_idx = _choose_available_indices(np.flatnonzero(remaining), null_count, rng)
    for idx in null_idx:
        target_column = rng.choice(["timestamp", "sensor_id", "reading", "status", "value"])
        df.at[idx, target_column] = np.nan
        df.at[idx, "type_valid"] = False
        df.at[idx, "anomaly_score"] = 0.25
        df.at[idx, "anomaly_type"] = "null"
        remaining[idx] = False

    type_mismatch_count = _rate_to_count(base_count, type_mismatch_rate)
    mismatch_idx = _choose_available_indices(
        np.flatnonzero(remaining),
        type_mismatch_count,
        rng,
    )
    for idx in mismatch_idx:
        mismatch_value = f"bad_reading_{idx}"
        df.at[idx, "reading"] = mismatch_value
        df.at[idx, "value"] = mismatch_value
        df.at[idx, "type_valid"] = False
        df.at[idx, "anomaly_score"] = 0.55
        df.at[idx, "anomaly_type"] = "type_mismatch"
        remaining[idx] = False

    malformed_date_count = _rate_to_count(base_count, malformed_date_rate)
    malformed_idx = _choose_available_indices(
        np.flatnonzero(remaining),
        malformed_date_count,
        rng,
    )
    invalid_dates = [
        "2026-13-40 25:61:00",
        "not_a_date",
        "2026/99/01",
        "31-02-2026",
    ]
    for pos, idx in enumerate(malformed_idx):
        bad_date = invalid_dates[pos % len(invalid_dates)]
        df.at[idx, "timestamp"] = bad_date
        df.at[idx, "type_valid"] = False
        df.at[idx, "anomaly_score"] = 0.7
        df.at[idx, "anomaly_type"] = "malformed_date"
        remaining[idx] = False

    drift_count = _rate_to_count(base_count, schema_drift_rate)
    drift_idx = _choose_available_indices(np.flatnonzero(remaining), drift_count, rng)
    for idx in drift_idx:
        drift_mode = rng.choice(["dict_value", "numeric_status", "list_sensor"])
        if drift_mode == "dict_value":
            drift_value = {"reading": df.at[idx, "reading"], "unit": "C"}
            df.at[idx, "value"] = drift_value
        elif drift_mode == "numeric_status":
            df.at[idx, "status"] = int(rng.integers(0, 3))
        else:
            df.at[idx, "sensor_id"] = [df.at[idx, "sensor_id"], "backup"]
        df.at[idx, "type_valid"] = False
        df.at[idx, "anomaly_score"] = 0.85
        df.at[idx, "anomaly_type"] = "schema_drift"
        remaining[idx] = False

    if duplicate_count > 0:
        duplicate_source_idx = rng.choice(df.index.to_numpy(), size=duplicate_count, replace=duplicate_count > len(df))
        duplicate_rows = df.loc[duplicate_source_idx].copy()
        duplicate_rows["anomaly_type"] = "duplicate"
        duplicate_rows["anomaly_score"] = np.maximum(
            duplicate_rows["anomaly_score"].astype(float).to_numpy(),
            0.4,
        )
        df = pd.concat([df, duplicate_rows], ignore_index=True)

    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def _serializable_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean_row = {}
        for key, value in row.items():
            if isinstance(value, float) and pd.isna(value):
                clean_row[key] = None
            else:
                clean_row[key] = value
        records.append(clean_row)
    return records


def generate_batch(
    task_id: str = "easy",
    seed: int = 42,
    task_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the batch mapping expected by ``NovaRLEnv._generate_batch``."""
    if task_id not in TASK_ANOMALIES:
        raise ValueError(f"Unknown task_id: {task_id}")

    task_config = {} if task_config is None else task_config
    batch_size = int(task_config.get("batch_size", 100))
    anomaly_rate = float(task_config.get("anomaly_rate", 0.10))
    enabled_anomalies = TASK_ANOMALIES[task_id]
    per_anomaly_rate = anomaly_rate / len(enabled_anomalies)

    df = generate_dirty_data(
        n_rows=batch_size,
        missing_rate=per_anomaly_rate if "null" in enabled_anomalies else 0.0,
        duplicate_rate=per_anomaly_rate if "duplicate" in enabled_anomalies else 0.0,
        type_mismatch_rate=(
            per_anomaly_rate if "type_mismatch" in enabled_anomalies else 0.0
        ),
        malformed_date_rate=(
            per_anomaly_rate if "malformed_date" in enabled_anomalies else 0.0
        ),
        schema_drift_rate=(
            per_anomaly_rate if "schema_drift" in enabled_anomalies else 0.0
        ),
        random_state=seed,
    )

    observed_counts = df["anomaly_type"].value_counts().to_dict()
    anomaly_counts = {
        anomaly_type: int(observed_counts.get(anomaly_type, 0))
        for anomaly_type in enabled_anomalies
    }
    sample_issue_summaries = [
        f"{name}: {count} rows affected"
        for name, count in anomaly_counts.items()
        if count > 0
    ][:3]

    return {
        "batch_size": len(df),
        "records": _serializable_records(df),
        "columns": list(df.columns),
        "anomaly_counts": anomaly_counts,
        "sample_issue_summaries": sample_issue_summaries,
    }


if __name__ == "__main__":
    sample = generate_dirty_data(n_rows=100)
    print("Anomaly counts:")
    print(sample["anomaly_type"].value_counts().sort_index())
    print("\nSample rows:")
    print(sample.head(10).to_string(index=False))
