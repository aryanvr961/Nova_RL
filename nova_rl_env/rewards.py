# =============================================================================
# File: nova_rl_env/rewards.py
# Owner: Aryan
# Role: Reward design and shaping
# =============================================================================
#
# Purpose
# - This file will define the reward signal used by the environment.
# - Reward shaping is one of the most important parts of the benchmark.
#
# What this file should contain later
# - Partial progress reward logic.
# - Penalties for unsafe promotion, over-quarantine, or wasteful actions.
# - Clean reward composition that is easy to explain.
#
# Design expectations
# - Rewards should not be purely binary.
# - Reward should correlate with meaningful task progress.
# - The final logic should help agents improve behavior over a trajectory.
#
# Constraints
# - Keep reward deterministic and interpretable.
# - Avoid hidden reward rules that are hard to debug.
