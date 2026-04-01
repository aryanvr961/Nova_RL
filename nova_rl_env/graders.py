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
