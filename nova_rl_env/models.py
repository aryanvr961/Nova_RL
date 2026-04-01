# =============================================================================
# File: nova_rl_env/models.py
# Owner: Aryan
# Role: Typed model definitions
# =============================================================================
#
# Purpose
# - This file will define the typed contract used by the environment.
# - It should contain the final Observation, Action, and Reward models.
#
# What this file should contain later
# - Observation model describing the current task and batch state.
# - Action model describing the structured decision taken by the agent.
# - Reward model describing the reward signal if the final environment uses typed rewards.
#
# Design expectations
# - Models should be deterministic, compact, and easy to validate.
# - Field names should reflect the actual environment semantics.
# - Avoid vague placeholder fields once implementation begins.
#
# Constraints
# - Use Pydantic models.
# - Keep the contract stable across inference, environment, and grading.
