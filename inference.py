# =============================================================================
# File: inference.py
# Owner: Aryan
# Role: Baseline evaluation entrypoint
# =============================================================================

from __future__ import annotations

import json
import os
from statistics import mean

from openai import OpenAI

from nova_rl_env.environment import NovaRLEnv
from nova_rl_env.models import Action, Observation


TASKS = ["easy", "medium", "hard"]
SEED = 42


def build_client() -> OpenAI:
    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("HF_TOKEN")
        or os.getenv("API_KEY")
        or "DUMMY_KEY"
    )
    base_url = os.getenv("API_BASE_URL")
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def observation_to_prompt(obs: Observation) -> str:
    return (
        "You are controlling a data quality remediation environment.\n"
        "Return strict JSON with keys: decision, threshold, notes.\n"
        "Allowed decisions: fix, quarantine, promote, noop, finalize.\n"
        f"Observation:\n{obs.model_dump_json(indent=2)}"
    )


def parse_action(text: str) -> Action:
    payload = json.loads(text)
    return Action(**payload)


def get_llm_action(client: OpenAI, obs: Observation) -> Action:
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    response = client.responses.create(
        model=model_name,
        input=observation_to_prompt(obs),
        temperature=0,
        max_output_tokens=120,
    )
    output_text = getattr(response, "output_text", "")
    return parse_action(output_text)


def main() -> None:
    client = build_client()
    env = NovaRLEnv()
    scores: list[float] = []

    for task_id in TASKS:
        env.set_task(task_id)
        obs = env.reset(seed=SEED)
        done = False
        final_info = {"grade": 0.0}

        while not done:
            action = get_llm_action(client, obs)
            obs, reward, done, info = env.step(action)
            _ = reward
            final_info = info

        score = float(final_info.get("grade", 0.0))
        scores.append(score)
        print(f"Task {task_id}: {score:.4f}")

    print(f"Final score: {mean(scores):.4f}")


if __name__ == "__main__":
    main()
