# =============================================================================
# File: nova_rl_env/environment.py
# Owner: Aryan
# Role: Main OpenEnv environment implementation
# =============================================================================
from __future__ import annotations

import importlib
import random
from typing import Any, Callable, Dict, Mapping, cast

from .models import Action, Observation, Reward
from .rewards import build_reward

DEFAULT_TASKS: Dict[str, Dict[str, Any]] = {
    "easy": {
        "objective": "Handle null values and exact duplicate rows safely.",
        "anomaly_rate": 0.05,
        "max_steps": 8,
    },
    "medium": {
        "objective": "Handle type mismatches and malformed date values.",
        "anomaly_rate": 0.15,
        "max_steps": 8,
    },
    "hard": {
        "objective": "Handle schema drift and correlated multi-column issues.",
        "anomaly_rate": 0.30,
        "max_steps": 8,
    },
}


class NovaRLEnv:
    """Lightweight OpenEnv-style environment with low-conflict extension points."""

    def __init__(self, task_id: str = "easy") -> None:
        self.task_id = task_id
        self.task_config = self._get_task_config(task_id)
        self.seed = 42
        self.step_index = 0
        self.current_threshold = 0.5
        self.last_action: str | None = None
        self.batch: Dict[str, Any] = {}
        self.current_metrics: Dict[str, float] = {}
        self.done = False

    def set_task(self, task_id: str) -> None:
        self.task_id = task_id
        self.task_config = self._get_task_config(task_id)

    def reset(self, seed: int | None = None) -> Observation:
        self.seed = 42 if seed is None else seed
        self.step_index = 0
        self.current_threshold = 0.5
        self.last_action = None
        self.done = False
        self.batch = self._generate_batch()
        self.current_metrics = {
            "fix_accuracy": 0.0,
            "promotion_precision": 0.0,
            "quarantine_precision": 0.0,
        }
        return self.state()

    def state(self) -> Observation:
        return Observation(
            task_id=self.task_id,  # type: ignore[arg-type]
            step_index=self.step_index,
            max_steps=int(self.task_config.get("max_steps", 8)),
            batch_size=int(self.batch.get("batch_size", 0)),
            anomaly_counts=dict(self.batch.get("anomaly_counts", {})),
            current_metrics=dict(self.current_metrics),
            sample_issue_summaries=list(self.batch.get("sample_issue_summaries", [])),
            current_threshold=self.current_threshold,
            remaining_steps=max(0, int(self.task_config.get("max_steps", 8)) - self.step_index),
            last_action=self.last_action,
        )

    def _resolve_action(self, action: Action | None) -> Action:
        if action is not None:
            return action
        return Action(
            decision="noop",
            threshold=self.current_threshold,
            notes="implicit_noop",
            parameters={},
        )

    def step(self, action: Action | None = None) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        resolved_action = self._resolve_action(action)
        if self.done:
            reward = build_reward(
                step_penalty=0.0,
                metadata={"reason": "episode_already_complete"},
            )
            return self.state(), reward, True, {
                "task_objective": self.task_config.get("objective", ""),
                "metrics": dict(self.current_metrics),
                "grade": self._clamp_open_score(self._grade(resolved_action)),
            }

        self.step_index += 1
        self.current_threshold = resolved_action.threshold
        self.last_action = resolved_action.decision

        progress = min(1.0, self.step_index / int(self.task_config.get("max_steps", 8)))
        self.current_metrics = self._estimate_metrics(resolved_action, progress)

        reward = build_reward(
            correct_fix_gain=self.current_metrics["fix_accuracy"] * 0.2,
            unsafe_promotion_penalty=(
                0.08
                if resolved_action.decision == "promote" and resolved_action.threshold < 0.45
                else 0.0
            ),
            over_quarantine_penalty=(
                0.08
                if resolved_action.decision == "quarantine" and resolved_action.threshold > 0.75
                else 0.0
            ),
            step_penalty=0.01,
            metadata={"task_id": self.task_id, "step_index": self.step_index},
        )

        done = (
            resolved_action.decision == "finalize"
            or self.step_index >= int(self.task_config.get("max_steps", 8))
        )
        self.done = done

        info = {
            "task_objective": self.task_config.get("objective", ""),
            "metrics": dict(self.current_metrics),
            "grade": self._clamp_open_score(self._grade(resolved_action)),
        }
        return self.state(), reward, done, info

    def _get_task_config(self, task_id: str) -> Dict[str, Any]:
        try:
            task_module = importlib.import_module("nova_rl_env.tasks")
            tasks_obj = getattr(task_module, "TASKS", None)
            if isinstance(tasks_obj, Mapping) and task_id in tasks_obj:
                cfg = tasks_obj[task_id]
                if isinstance(cfg, Mapping):
                    return dict(cfg)
                if hasattr(cfg, "__dict__"):
                    return dict(vars(cfg))
            get_task = getattr(task_module, "get_task_config", None)
            if callable(get_task):
                cfg = get_task(task_id)
                if isinstance(cfg, Mapping):
                    return dict(cfg)
        except (ImportError, AttributeError, KeyError, TypeError, ValueError):
            pass
        return dict(DEFAULT_TASKS[task_id])

    def _generate_batch(self) -> Dict[str, Any]:
        try:
            datagen_module = importlib.import_module("nova_rl_env.datagen")
            generate_batch = getattr(datagen_module, "generate_batch", None)
            if callable(generate_batch):
                batch = generate_batch(
                    task_id=self.task_id,
                    seed=self.seed,
                    task_config=self.task_config,
                )
                if isinstance(batch, Mapping):
                    return dict(batch)
        except (ImportError, AttributeError, KeyError, TypeError, ValueError):
            pass

        rng = random.Random(self.seed)
        batch_size = 100
        anomaly_rate = float(self.task_config.get("anomaly_rate", 0.1))
        anomaly_budget = max(1, int(batch_size * anomaly_rate))
        anomaly_types = {
            "easy": ["null", "duplicate"],
            "medium": ["null", "duplicate", "type_mismatch", "malformed_date"],
            "hard": [
                "null",
                "duplicate",
                "type_mismatch",
                "malformed_date",
                "schema_drift",
            ],
        }[self.task_id]
        anomaly_counts = {name: 0 for name in anomaly_types}
        for _ in range(anomaly_budget):
            anomaly_counts[rng.choice(anomaly_types)] += 1
        return {
            "batch_size": batch_size,
            "anomaly_counts": anomaly_counts,
            "sample_issue_summaries": [
                f"{name}: {count} rows affected"
                for name, count in anomaly_counts.items()
                if count > 0
            ][:3],
        }

    def _estimate_metrics(self, action: Action, progress: float) -> Dict[str, float]:
        if action.decision == "fix":
            return {
                "fix_accuracy": min(1.0, 0.35 + 0.55 * progress),
                "promotion_precision": min(1.0, 0.25 + 0.35 * progress),
                "quarantine_precision": min(1.0, 0.20 + 0.30 * progress),
            }
        if action.decision == "quarantine":
            return {
                "fix_accuracy": min(1.0, 0.15 + 0.20 * progress),
                "promotion_precision": min(1.0, 0.15 + 0.20 * progress),
                "quarantine_precision": min(1.0, 0.40 + 0.45 * progress),
            }
        if action.decision == "promote":
            return {
                "fix_accuracy": min(1.0, 0.10 + 0.20 * progress),
                "promotion_precision": min(1.0, 0.35 + 0.45 * progress),
                "quarantine_precision": min(1.0, 0.05 + 0.10 * progress),
            }
        return {
            "fix_accuracy": min(1.0, 0.05 + 0.10 * progress),
            "promotion_precision": min(1.0, 0.05 + 0.10 * progress),
            "quarantine_precision": min(1.0, 0.05 + 0.10 * progress),
        }

    def _clamp_open_score(self, value: float) -> float:
        min_score = 0.01
        max_score = 0.99
        return max(min_score, min(max_score, float(value)))

    def _grade(self, action: Action) -> float:
        try:
            graders_module = importlib.import_module("nova_rl_env.graders")
            grade_fn = getattr(graders_module, "grade", None)
            if callable(grade_fn):
                typed_grade_fn = cast(
                    Callable[..., float],
                    grade_fn,
                )
                grading_state: Dict[str, Any] = {
                    "batch": self.batch,
                    "metrics": self.current_metrics,
                    "step_index": self.step_index,
                }
                return self._clamp_open_score(
                    typed_grade_fn(
                        task_id=self.task_id,
                        state=grading_state,
                        action=action,
                    )
                )
        except (ImportError, AttributeError, TypeError, ValueError):
            pass

        score = (
            0.6 * self.current_metrics.get("fix_accuracy", 0.0)
            + 0.2 * self.current_metrics.get("promotion_precision", 0.0)
            + 0.2 * self.current_metrics.get("quarantine_precision", 0.0)
        )
        return self._clamp_open_score(score)
