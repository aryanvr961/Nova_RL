# Nova RL

Nova RL is an OpenEnv-style ETL data remediation environment being prepared for hackathon submission. The project is centered on a real-world workflow where an agent reviews noisy tabular batches, takes structured remediation decisions, and is evaluated through deterministic rewards and grading.

## Current Status

The repository is no longer only a file scaffold. It now contains:

- a typed model layer
- a reusable reward layer
- a lightweight environment skeleton
- a baseline inference flow using the OpenAI client
- deployment-facing files for container and HF startup

Task definitions, data generation, and task-specific grading are still being completed and integrated.

## Problem Setting

Modern ETL pipelines frequently fail because of:

- missing values
- duplicate records
- malformed dates
- type mismatches
- schema drift

Nova RL turns that operational cleanup workflow into an agent benchmark with a standard environment contract.

## Environment Scope

- Domain: ETL data quality remediation
- Interface: `reset()`, `step(action)`, `state()`
- Difficulty levels: `easy`, `medium`, `hard`
- Typed models: `Observation`, `Action`, `Reward`

## Implemented Foundation

### Models

The repository already includes typed models for:

- `Observation`
- `Action`
- `Reward`

These provide the shared contract across the environment, inference, and reward layers.

### Reward Layer

The reward module already includes:

- reusable reward composition helpers
- support for dense step-wise reward signals
- explicit penalty channels for unsafe promotion, over-quarantine, and step cost

### Environment Base

The current environment implementation already includes:

- task switching
- deterministic reset flow
- observation generation
- step transitions
- fallback task config loading
- fallback batch generation
- fallback grade computation

This is intended to support low-conflict parallel development while task-specific logic is being finalized.

### Baseline Inference

The current inference flow already includes:

- OpenAI client initialization
- environment-variable-based configuration
- task loop for `easy`, `medium`, and `hard`
- observation-to-prompt conversion
- structured action parsing

## Observation Space

The current observation contract is designed to expose:

- task identifier
- current step index
- max steps
- batch size
- anomaly counts by type
- current metrics
- issue summaries
- current threshold
- remaining steps
- previous action summary

## Action Space

The current action contract supports:

- a bounded threshold value
- a structured decision type
- optional notes
- optional extensible parameters

Current decision set:

- `fix`
- `quarantine`
- `promote`
- `noop`
- `finalize`

## Task Design

Nova RL is planned around three required task levels:

- Easy: handle null values and exact duplicates safely
- Medium: handle type mismatches and malformed dates
- Hard: handle schema drift and correlated multi-column issues

Each task is expected to represent a concrete objective and produce deterministic scores in the range `0.0` to `1.0`.

## Repository Structure

```text
NOVA_RL/
├── .env.example
├── .gitignore
├── app.py
├── Dockerfile
├── inference.py
├── NovaRL_Final_Roadmap.docx
├── openenv.yaml
├── README.md
├── requirements.txt
└── nova_rl_env/
    ├── __init__.py
    ├── datagen.py
    ├── environment.py
    ├── graders.py
    ├── models.py
    ├── rewards.py
    └── tasks.py
```

## Setup

Install project dependencies:

```bash
pip install -r requirements.txt
```

## Baseline Inference

The final baseline is expected to:

- use the OpenAI client for LLM calls
- read credentials from runtime environment variables
- run all three tasks
- print per-task scores and a final aggregate score
- remain reproducible under fixed seeds

## Environment Variables

Depending on runtime setup, the project may use:

- `OPENAI_API_KEY`
- `HF_TOKEN`
- `API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`

For local development, copy `.env.example` into `.env` and fill in the required values. For judge or HF execution, runtime-injected environment variables are expected.

## Deployment

The repository includes deployment-facing files required for packaging:

- `Dockerfile`
- `openenv.yaml`
- `app.py`

The current deployment layer is still in MVP stage and should be verified through final container and HF testing before submission.

## Pending Integration Work

The following pieces still need final task-specific implementation or alignment:

- `tasks.py`
- `datagen.py`
- `graders.py`
- final environment-to-grader contract
- final environment-to-datagen contract
- final `openenv.yaml` validation
- final Docker and HF verification

## Baseline Scores

Baseline scores will be recorded after full task integration. The final README should include:

- Easy score
- Medium score
- Hard score
- Final mean score

## Ownership

- Aryan: environment, models, rewards, inference, deployment-facing files
- Aadyaa: tasks, data generation, grading logic, and task-facing documentation

## Submission Goal

The final deliverable should be lightweight, deterministic, compliant with the OpenEnv interface, and cleanly deployable for hackathon evaluation.
