from __future__ import annotations

import json
import logging
import os
from typing import Optional

from nova_rl_env.config import load_env_file

load_env_file()

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from nova_rl_env.environment import NovaRLEnv
from nova_rl_env.llm import (
    build_llm_client,
    fallback_action,
    format_llm_label,
    get_llm_action,
    is_fatal_llm_error,
    resolve_llm_config,
)
from nova_rl_env.memory import (
    generate_session_id,
    record_episode_end_async,
    record_session_start_async,
    record_step_async,
    shutdown_memory_writer,
    truncate_text,
)
from nova_rl_env.models import LLMConfig


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


def _single_line(value: object) -> str:
    return str(value).replace("\r", " ").replace("\n", " ").strip()


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
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.6f} rewards={rewards_str}",
        flush=True,
    )


def clamp_open_score(value: float) -> float:
    return max(MIN_SCORE, min(MAX_SCORE, value))


def sanitize_action_text(action: object) -> str:
    model_dump = getattr(action, "model_dump", None)
    if callable(model_dump):
        return json.dumps(model_dump(exclude_none=True), separators=(",", ":"))
    return json.dumps(action, separators=(",", ":"))


def close_env(env: NovaRLEnv) -> None:
    close_fn = getattr(env, "close", None)
    if callable(close_fn):
        close_fn()


def run_task(client: object, llm_config: LLMConfig, task_id: str) -> None:
    env = NovaRLEnv(task_id=task_id)
    session_id = generate_session_id(task_id)
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_error: str | None = None
    llm_disabled_error: str | None = None

    log_start(
        task=task_id,
        env=BENCHMARK,
        model=format_llm_label(llm_config),
        session_id=session_id,
    )
    logger.info("Starting task=%s session=%s llm=%s", task_id, session_id, format_llm_label(llm_config))

    try:
        obs = env.reset(seed=SEED)
        record_session_start_async(
            session_id=session_id,
            task_id=task_id,
            seed=env.seed,
            observation=obs,
            llm_provider=llm_config.provider,
            llm_model=llm_config.model,
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
                    action = get_llm_action(client, llm_config, obs)
                except Exception as exc:
                    error = truncate_text(_single_line(exc), 240)
                    logger.error("LLM call failed provider=%s model=%s error=%s", llm_config.provider, llm_config.model, error)
                    if is_fatal_llm_error(error):
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
        logger.exception("Task failed session=%s error=%s", session_id, final_error)
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
    llm_config = resolve_llm_config()
    client = build_llm_client(llm_config)
    for task_id in TASKS:
        run_task(client, llm_config, task_id)


if __name__ == "__main__":
    main()
