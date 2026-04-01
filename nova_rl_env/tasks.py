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
