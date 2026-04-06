---
title: Nova RL
emoji: "🤖"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# Nova RL

Nova RL is an OpenEnv environment for ETL data remediation. The agent receives noisy tabular batches, takes structured remediation actions, and is scored with deterministic graders across `easy`, `medium`, and `hard` tasks.

## Environment Summary

- Domain: ETL data quality remediation
- Interface: typed `reset()`, `step(action)`, and `state()`
- Tasks: `easy`, `medium`, `hard`
- Score range: deterministic `0.0` to `1.0`

## Real-World Task

The benchmark simulates common ETL cleanup failures:

- missing values
- duplicate rows
- malformed dates
- type mismatches
- schema drift

## Observation Space

The observation payload exposes:

- `task_id`
- `step_index`
- `max_steps`
- `batch_size`
- `anomaly_counts`
- `current_metrics`
- `sample_issue_summaries`
- `current_threshold`
- `remaining_steps`
- `last_action`

## Action Space

The agent can return:

- `fix`
- `quarantine`
- `promote`
- `noop`
- `finalize`

Each action includes a bounded `threshold`, optional `notes`, and optional structured `parameters`.

## Task Definitions

- Easy: handle null values and exact duplicate rows safely
- Medium: handle type mismatches and malformed dates
- Hard: handle schema drift and correlated multi-column issues

Task configs live in `nova_rl_env/tasks.py`, synthetic batch generation lives in `nova_rl_env/datagen.py`, and deterministic grading lives in `nova_rl_env/graders.py`.

## Repository Layout

```text
NOVA_RL/
|-- app.py
|-- Dockerfile
|-- inference.py
|-- openenv.yaml
|-- README.md
|-- requirements.txt
`-- nova_rl_env/
    |-- datagen.py
    |-- environment.py
    |-- graders.py
    |-- models.py
    |-- rewards.py
    `-- tasks.py
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and define the inference variables before running `inference.py`.

## Runtime API

The FastAPI app exposes:

- `GET /ping`
- `GET /health`
- `GET /reset`
- `GET /state`
- `POST /step`

`/reset` creates or resets a session-scoped environment and returns a `session_id`. `/state` and `/step` require that `session_id`.

## Inference Configuration

The baseline inference script uses the OpenAI Python client with environment variables aligned to the requirement doc:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Compatibility fallbacks also supported:

- `OPENAI_API_KEY`
- `API_KEY`

By default the script targets the Hugging Face router endpoint and reads `MODEL_NAME` from the environment.

## Deployment

The project is packaged for Hugging Face Space deployment:

- `Dockerfile`
- `openenv.yaml`
- `app.py`

`openenv.yaml` uses an OpenEnv-style FastAPI Space manifest with `spec_version`, `runtime`, `app`, and `port`.

## Baseline Scores

Representative task/grader integration checks produced valid scores in the required range:

- Easy: `0.8`
- Medium: `0.5375`
- Hard: `0.4725`

## Ownership

- Aryan: environment, models, rewards, inference, deployment-facing files
- Aadyaa: tasks, data generation, grading logic, and task-facing documentation
