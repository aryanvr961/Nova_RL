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
