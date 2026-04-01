# =============================================================================
# File: Dockerfile
# Owner: Aryan
# Role: Container build specification
# =============================================================================
#
# Purpose
# - This file will define how the project is built and run inside a container.
# - It is required for packaging and likely required for HF deployment.
#
# What this file should contain later
# - Base Python image.
# - Dependency installation steps.
# - Project copy steps.
# - Final runtime command.
#
# Constraints
# - The image must build cleanly.
# - The container must start cleanly.
# - Keep the image as small and simple as possible.
# - Do not rely on machine-specific local state.
