from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from nova_rl_env.environment import NovaRLEnv
from nova_rl_env.models import Action, Observation


TASKS = ["easy", "medium", "hard"]
SEED = 42
BENCHMARK = os.getenv("NOVA_RL_BENCHMARK", "nova_rl")
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.1


def get_api_base_url() -> str:
    return os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")


def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "openai/gpt-4o-mini")


def get_api_key() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_client() -> OpenAI:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "Missing API credentials. Set HF_TOKEN or API_KEY before running inference.py."
        )
    return OpenAI(api_key=api_key, base_url=get_api_base_url())


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


def sanitize_action_text(action: Action) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


def fallback_action(obs: Observation, error: Optional[str] = None) -> Action:
    if obs.remaining_steps <= 1:
        note = "fallback_finalize"
        if error:
            note = f"{note}:{error}"
        return Action(decision="finalize", threshold=0.5, notes=note, parameters={})

    note = "fallback_fix"
    if error:
        note = f"{note}:{error}"
    return Action(decision="fix", threshold=0.5, notes=note, parameters={})


def get_llm_action(client: OpenAI, obs: Observation) -> Action:
    response = client.responses.create(
        model=get_model_name(),
        input=observation_to_prompt(obs),
        temperature=0,
        max_output_tokens=120,
    )
    output_text = getattr(response, "output_text", "")
    if not output_text:
        raise RuntimeError("Model returned empty output_text; cannot parse action.")
    return parse_action(output_text)


def close_env(env: NovaRLEnv) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def run_task(client: OpenAI, task_id: str) -> None:
    env = NovaRLEnv(task_id=task_id)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=get_model_name())

    try:
        obs = env.reset(seed=SEED)
        done = False
        final_info: dict[str, object] = {"grade": 0.0}

        while not done and steps_taken < MAX_STEPS:
            error: Optional[str] = None

            try:
                action = get_llm_action(client, obs)
            except Exception as exc:
                error = str(exc).replace("\n", " ")
                action = fallback_action(obs, error=error)

            obs, reward, done, info = env.step(action)
            reward_value = float(reward.value)
            rewards.append(reward_value)
            steps_taken += 1
            final_info = info

            log_step(
                step=steps_taken,
                action=sanitize_action_text(action),
                reward=reward_value,
                done=done,
                error=error,
            )

        score = float(final_info.get("grade", 0.0))
        score = min(max(score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        try:
            close_env(env)
        finally:
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = build_client()
    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
