# =============================================================================
# File: inference.py
# Owner: Aryan
# Role: Baseline evaluation entrypoint
# =============================================================================
#
# Purpose
# - This is the mandatory root-level baseline script for hackathon submission.
# - It will run the model against the environment on all required tasks.
#
# What this file should contain later
# - Environment initialization.
# - Task loop for easy, medium, and hard.
# - Observation-to-prompt conversion.
# - OpenAI-compatible model call.
# - Parsing of model output into a structured Action.
# - Passing the Action into env.step(action).
# - Final score reporting for each task and overall mean score.
#
# Runtime requirements
# - File name must remain exactly `inference.py`.
# - Must use the OpenAI client for all LLM calls.
# - Must read runtime configuration from environment variables.
# - Must be reproducible under fixed seeds.
# - Must complete within the hackathon runtime limit.
#
# Important guardrails
# - Do not hardcode model names, API keys, or endpoints.
# - Do not replace the agent with a purely rule-based policy in the final version.
# - Fallback behavior should be safe, but the primary path must use an actual LLM call.
