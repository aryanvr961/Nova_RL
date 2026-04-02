# =============================================================================
# File: nova_rl_env/models.py
# Owner: Aryan
# Role: Typed model definitions
# =============================================================================
#
# Purpose
# - Defines the stable typed contract shared across the environment, inference,
#   and grading layers.
# - Keeps the MVP schema compact, explicit, and easy to validate.
#
# Notes
# - These models are intentionally generic enough to avoid conflict with
#   task-specific work in `tasks.py`, `datagen.py`, and `graders.py`.
# - Task and anomaly details can evolve without forcing repeated schema changes.

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


TaskId = Literal["easy", "medium", "hard"]
DecisionType = Literal["fix", "quarantine", "promote", "noop", "finalize"]


class Observation(BaseModel):
    """Structured state returned to the agent on every step."""

    task_id: TaskId
    step_index: int = Field(ge=0, description="Current step within the episode.")
    max_steps: int = Field(gt=0, description="Maximum allowed steps in the episode.")
    batch_size: int = Field(ge=0, description="Number of records in the current batch.")
    anomaly_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of affected rows by anomaly type.",
    )
    current_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Rolling environment metrics visible to the agent.",
    )
    sample_issue_summaries: List[str] = Field(
        default_factory=list,
        description="Compact human-readable summaries of current issues.",
    )
    current_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Current remediation or promotion threshold.",
    )
    remaining_steps: int = Field(
        ge=0,
        description="How many steps remain before the episode ends.",
    )
    last_action: Optional[str] = Field(
        default=None,
        description="Action taken in the previous step, if any.",
    )


class Action(BaseModel):
    """Structured decision produced by the agent."""

    decision: DecisionType = Field(
        description="Primary action to apply in the current step."
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Control threshold used by the environment.",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional compact rationale or summary from the agent.",
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured parameters for future extensibility.",
    )


class Reward(BaseModel):
    """Typed reward payload returned by the environment."""

    value: float = Field(description="Primary scalar reward for the step.")
    correct_fix_gain: float = Field(
        default=0.0,
        description="Positive contribution from correct remediation progress.",
    )
    unsafe_promotion_penalty: float = Field(
        default=0.0,
        description="Penalty for promoting bad or uncertain rows.",
    )
    over_quarantine_penalty: float = Field(
        default=0.0,
        description="Penalty for quarantining too aggressively.",
    )
    step_penalty: float = Field(
        default=0.0,
        description="Small penalty to discourage wasteful extra steps.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional reward diagnostics.",
    )
