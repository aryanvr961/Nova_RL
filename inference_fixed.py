from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any, Optional

from nova_rl_env.config import load_env_file

load_env_file()

from google import genai
from google.genai import types

from nova_rl_env.environment import NovaRLEnv
from nova_rl_env.memory import (
    generate_session_id,
    record_episode_end_async,
    record_session_start_async,
    record_step_async,
    shutdown_memory_writer,
    truncate_text,
)
from nova_rl_env.models import Action, Observation


TASKS = [
    task.strip()
    for task in os.getenv("NOVA_RL_TASKS", "easy,medium,hard").split(",")
    if task.strip()
]
SEED = 42
BENCHMARK = os.getenv("NOVA_RL_BENCHMARK", "nova_rl")
MAX_STEPS = int(os.getenv("NOVA_RL_MAX_STEPS", "8"))
SUCCESS_SCORE_THRESHOLD = 0.1
MIN_SCORE = 0.01
MAX_SCORE = 0.99
GEMINI_TIMEOUT_SECONDS = 15.0
GEMINI_MAX_RETRIES = 1
FATAL_GEMINI_ERROR_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "429",
    "NOT_FOUND",
    "404",
)


def get_llm_provider() -> str:
    return "gemini"


def get_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_api_key() -> str | None:
    return os.getenv("GEMINI_API_KEY")


def _single_line(value: Any) -> str:
    return str(value).replace("\\r", " ").replace("\\n", " ").strip()


def log_start(task: str, env: str, model: str, session_id: str) -> None:
    print(
        f"[START] task={task} env={env} model={model} session_id={session_id}",
        flush=True,
    )


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = truncate_text(_single_line(error), 240) if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={_single_line(action)} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


def clamp_open_score(value: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, value))


def build_client() -> Any:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY missing. Set it in .env to enable Gemini inference."
        )
    return genai.Client(api_key=api_key)


def observation_to_prompt(obs: Observation) -> str:
    observation_json = obs.model_dump_json(exclude_none=True)
    return (
        "Return only JSON for an ETL remediation action. "
        "Schema: {\\\"decision\\\":\\\"fix|quarantine|promote|noop|finalize\\\","
        "\\\"threshold\\\":0.5,\\\"notes\\\":\\\"short reason\\\"}. "
        f"Observation: {observation_json}"
    )


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :] .strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1].strip()
    return cleaned


def parse_action(text: str) -> Action:
    cleaned = strip_code_fences(text)
    if not cleaned:
        raise ValueError("Gemini returned empty output.")
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError("Gemini output must be a JSON object.")
    if "decision" not in payload:
        raise ValueError("Gemini output missing required decision field.")
    return Action(**payload)


def sanitize_action_text(action: Action) -> str:
    return json.dumps(action.model_dump(exclude_none=True), separators=(",", ":"))


def fallback_action(obs: Observation, error: Optional[str] = None) -> Action:
    suffix = truncate_text(error, 220)
    if obs.remaining_steps <= 1:
        note = "fallback_finalize"
        if suffix:
            note = f"{note}:{suffix}"
        return Action(decision="finalize", threshold=0.5, notes=note, parameters={})

    note = "fallback_fix_safe"
    if suffix:
        note = f"{note}:{suffix}"
    return Action(decision="fix", threshold=0.5, notes=note, parameters={})


def _generate_content_once(client: genai.Client, obs: Observation) -> str:
    response = client.models.generate_content(
        model=get_model_name(),
        contents=observation_to_prompt(obs),
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0,
            max_output_tokens=120,
            thinking_config=types.ThinkingConfig(
                include_thoughts=False,
                thinking_budget=0,
            ),
        ),
    )
    output_text = (response.text or "").strip()
    if not output_text:
        raise RuntimeError("Gemini returned empty output_text; cannot parse action.")
    return output_text


def _generate_content_with_timeout(client: genai.Client, obs: Observation) -> str:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_generate_content_once, client, obs)
    try:
        return future.result(timeout=GEMINI_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            f"Gemini call timed out after {GEMINI_TIMEOUT_SECONDS:.0f}s."
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def get_llm_action(client: Any, obs: Observation) -> Action:
    last_error: Exception | None = None
    max_retries = GEMINI_MAX_RETRIES
    for attempt in range(max_retries + 1):
        try:
            return parse_action(_generate_content_with_timeout(client, obs))
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break
    raise RuntimeError(
        truncate_text(last_error, 240) or "Gemini action generation failed."
    )


def is_fatal_gemini_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.upper()
    return any(marker in normalized for marker in FATAL_GEMINI_ERROR_MARKERS)


def close_env(env: NovaRLEnv) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def run_task(client: Any, task_id: str) -> None:
    env = NovaRLEnv(task_id=task_id)
    session_id = generate_session_id(task_id)
    rewards = []
    steps_taken = 0
    score = 0.0
    success = False
    final_error: Optional[str] = None
    llm_disabled_error: Optional[str] = None

    log_start(
        task=task_id,
        env=BENCHMARK,
        model=f"gemini:{get_model_name()}",
        session_id=session_id,
    )

    try:
        obs = env.reset(seed=SEED)
        record_session_start_async(
            session_id=session_id,
            task_id=task_id,
            seed=env.seed,
            observation=obs,
        )
        done = False
        final_info: dict[str, float] = {"grade": 0.0}

        while not done and steps_taken < MAX_STEPS:
            error: Optional[str] = None

            if llm_disabled_error:
                error = llm_disabled_error
                action = fallback_action(obs, error=error)
            else:
                try:
                    action = get_llm_action(client, obs)
                except Exception as exc:
                    error = truncate_text(_single_line(exc), 240)
                    if is_fatal_gemini_error(error):
                        llm_disabled_error = error
                    action = fallback_action(obs, error=error)

            obs, reward, done, info = env.step(action)
            reward_value = float(reward.value)
            rewards.append(reward_value)
            steps_taken += 1
            final_info = info
            grade = clamp_open_score(float(info.get("grade") or 0.0))

            record_step_async(
                session_id=session_id,
                task_id=task_id,
                step_index=steps_taken,
                observation=obs,
                action=action,
                reward=reward,
                done=done,
                grade=grade,
                error=error,
            )

            log_step(
                step=steps_taken,
                action=sanitize_action_text(action),
                reward=reward_value,
                done=done,
                error=error,
            )

        score = clamp_open_score(float(final_info.get("grade") or 0.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
    except Exception as exc:
        final_error = truncate_text(_single_line(exc), 240)
        raise
    finally:
        try:
            close_env(env)
        finally:
            score = clamp_open_score(score)
            record_episode_end_async(
                session_id=session_id,
                task_id=task_id,
                steps=steps_taken,
                score=score,
                success=success,
                rewards=rewards,
                error=final_error,
            )
            shutdown_memory_writer()
            log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    client = build_client()
    for task_id in TASKS:
        run_task(client, task_id)


if __name__ == "__main__":
    main()
