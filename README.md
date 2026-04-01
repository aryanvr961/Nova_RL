# Nova RL

Nova RL is an OpenEnv project scaffold for a real-world ETL data remediation environment. The benchmark is designed around structured data quality workflows where an LLM agent reviews noisy tabular batches, chooses remediation actions, and is evaluated through deterministic grading.

## Project Status

This repository is currently in scaffold stage. The file structure, ownership boundaries, and submission-facing components are in place, while the environment logic and baseline implementation are intended to be completed next.

## Problem Setting

Modern data pipelines often fail because of issues such as missing values, duplicate records, malformed dates, schema drift, and type inconsistencies. Nova RL turns that operational workflow into an agent benchmark that can be evaluated through a standard environment interface.

## Environment Scope

- Domain: ETL data quality remediation
- Interface: `reset()`, `step(action)`, `state()`
- Difficulty levels: `easy`, `medium`, `hard`
- Typed models: `Observation`, `Action`, `Reward`

## Observation Space

The observation design is intended to describe the current batch and environment state in a structured form. The final implementation is expected to include:

- task identifier
- current step index
- batch size
- anomaly counts by type
- issue summaries or sample diagnostics
- current thresholds or policy state
- remaining step budget
- previous action summary

## Action Space

The action design is intended to remain compact and structured for reliable evaluation. The final implementation is expected to include:

- a bounded threshold value
- a decision type such as fix, quarantine, promote, noop, or finalize

## Task Design

Nova RL is planned around three required task levels:

- Easy: handle null values and exact duplicates safely
- Medium: handle type mismatches and malformed dates
- Hard: handle schema drift and correlated multi-column issues

Each task should represent a concrete objective, use deterministic grading, and return a score in the range `0.0` to `1.0`.

## Reward and Evaluation

The final environment is expected to use a dense reward strategy rather than a purely binary success signal. Reward shaping should reflect meaningful progress while penalizing unsafe promotion, over-quarantine, or wasteful actions.

Evaluation should remain deterministic and reproducible across runs.

## Repository Structure

```text
NOVA_RL/
├── .gitignore
├── app.py
├── Dockerfile
├── inference.py
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

The final baseline must be implemented in root-level `inference.py` and should:

- use the OpenAI client for LLM calls
- read credentials from environment variables
- run all three tasks
- print per-task scores and a final aggregate score
- remain reproducible under fixed seeds

## Environment Variables

Depending on the final inference setup, the project may use:

- `OPENAI_API_KEY`
- `HF_TOKEN`
- `API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`

## Deployment

The repository includes the standard submission-facing files required for packaging and deployment:

- `Dockerfile`
- `openenv.yaml`
- `app.py`

These files are intended to support local containerization and Hugging Face Space deployment once implementation is completed.

## Baseline Scores

Baseline scores will be recorded after implementation. The final README should include:

- Easy score
- Medium score
- Hard score
- Final mean score

## Ownership

- Aryan: environment, models, rewards, inference, deployment-facing files
- Aadyaa: tasks, data generation, grading logic, and task-facing documentation

## Submission Goal

The final deliverable should be lightweight, deterministic, compliant with the OpenEnv interface, and cleanly deployable for hackathon evaluation.
