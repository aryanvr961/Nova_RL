from __future__ import annotations

import os
import re
from concurrent.futures import Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Mapping
from uuid import uuid4

from .config import load_env_file


load_env_file()

ERROR_TEXT_LIMIT = 240
MAX_WORKERS = 2
MAX_PENDING_WRITES = int(os.getenv("NOVA_RL_MAX_PENDING_WRITES", "100"))
DEFAULT_COLLECTION = "nova_rl_sessions"

_EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS, thread_name_prefix="nova-memory")
_LOCK = Lock()
_PENDING_WRITES = 0
_DROPPED_WRITES = 0
_FUTURES: set[Future[Any]] = set()
_CLIENT: Any | None = None
_INIT_ATTEMPTED = False
_INIT_ERROR: str | None = None


def _firebase_configured() -> bool:
    return any(
        os.getenv(name)
        for name in (
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_CLOUD_PROJECT",
            "GCLOUD_PROJECT",
            "FIREBASE_CONFIG",
            "K_SERVICE",
        )
    )


def truncate_text(value: Any, limit: int = ERROR_TEXT_LIMIT) -> str | None:
    if value is None:
        return None
    text = str(value).replace("\r", " ").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return f"{text[: max(0, limit - 3)]}..."


def generate_session_id(task_id: str) -> str:
    safe_task = re.sub(r"[^a-zA-Z0-9_-]+", "-", task_id).strip("-") or "task"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{safe_task}_{timestamp}_{uuid4().hex[:6]}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _collection_name() -> str:
    return os.getenv("NOVA_RL_FIRESTORE_COLLECTION", DEFAULT_COLLECTION)


def _get_client() -> Any | None:
    global _CLIENT, _INIT_ATTEMPTED, _INIT_ERROR
    if _INIT_ATTEMPTED:
        return _CLIENT

    _INIT_ATTEMPTED = True
    if not _firebase_configured():
        _INIT_ERROR = "firebase_not_configured"
        return None

    try:
        import firebase_admin
        from firebase_admin import firestore

        if not firebase_admin._apps:
            firebase_admin.initialize_app()
        _CLIENT = firestore.client()
    except Exception as exc:
        _CLIENT = None
        _INIT_ERROR = truncate_text(exc) or "firebase_init_failed"
    return _CLIENT


def _to_mapping(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _extract_anomalies(observation: Any) -> dict[str, int]:
    """Extract anomaly counts from observation for demo and audit logs."""
    data = _to_mapping(observation)
    anomaly_counts = data.get("anomaly_counts", {})
    if not isinstance(anomaly_counts, dict):
        return {}
    clean_counts: dict[str, int] = {}
    for key, value in anomaly_counts.items():
        try:
            clean_counts[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return clean_counts


def _extract_summary(observation: Any) -> str:
    """Extract and join sample issue summaries into a single readable string."""
    data = _to_mapping(observation)
    summaries = data.get("sample_issue_summaries", [])
    if not summaries or not isinstance(summaries, list):
        return "no issues detected"
    # Join summaries, truncate if necessary
    summary_text = " | ".join(str(s).strip() for s in summaries if s)
    return summary_text or "no issues detected"


def _extract_decision(action: Any) -> str:
    """Extract decision field from action."""
    data = _to_mapping(action)
    decision = data.get("decision", "unknown")
    return str(decision).strip() or "unknown"


def _extract_reward_value(reward: Any) -> float:
    """Extract reward value from reward object."""
    data = _to_mapping(reward)
    value = data.get("value", 0.0)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _write_session_start(
    *,
    session_id: str,
    task_id: str,
    seed: int | None,
    observation: Any,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> None:
    client = _get_client()
    if client is None:
        return
    now = _now_iso()
    client.collection(_collection_name()).document(session_id).set(
        {
            "task_id": task_id,
            "status": "running",
            "started_at": now,
            "steps": 0,
            "latest_decision": None,
            "llm_provider": llm_provider,
            "llm_model": llm_model,
        }
    )


def _write_step(
    *,
    session_id: str,
    task_id: str,
    step_index: int,
    observation: Any,
    action: Any,
    reward: Any,
    done: bool,
    grade: float | None,
    error: str | None,
) -> None:
    client = _get_client()
    if client is None:
        return
    
    now = _now_iso()
    session_ref = client.collection(_collection_name()).document(session_id)
    
    # Extract minimal data for step document
    decision = _extract_decision(action)
    reward_value = _extract_reward_value(reward)
    anomalies = _extract_anomalies(observation)
    summary = _extract_summary(observation)
    
    # Write minimal step document
    step_payload = {
        "step_index": step_index,
        "decision": decision,
        "reward": reward_value,
        "anomalies": anomalies,
        "summary": summary,
        "timestamp": now,
    }
    session_ref.collection("steps").document(f"step_{step_index:03d}").set(step_payload)
    
    # Update session with latest_decision and steps count
    session_update = {
        "latest_decision": decision,
        "steps": step_index,
    }
    
    # If episode is done, set completion fields
    if done:
        session_update["status"] = "complete"
        session_update["completed_at"] = now
        if grade is not None:
            session_update["final_grade"] = grade
    
    session_ref.set(session_update, merge=True)


def _write_episode_end(
    *,
    session_id: str,
    task_id: str,
    steps: int,
    score: float,
    success: bool,
    rewards: list[float],
    error: str | None,
) -> None:
    client = _get_client()
    if client is None:
        return
    
    now = _now_iso()
    client.collection(_collection_name()).document(session_id).set(
        {
            "status": "complete",
            "completed_at": now,
            "final_grade": score,
            "steps": steps,
        },
        merge=True,
    )


def _schedule(write_fn: Any, **kwargs: Any) -> Future[Any] | None:
    global _DROPPED_WRITES, _PENDING_WRITES
    with _LOCK:
        if _PENDING_WRITES >= MAX_PENDING_WRITES:
            _DROPPED_WRITES += 1
            return None
        _PENDING_WRITES += 1

    def wrapped() -> None:
        global _PENDING_WRITES
        try:
            write_fn(**kwargs)
        except Exception:
            # Firestore memory is best-effort and must never break episodes.
            pass
        finally:
            with _LOCK:
                _PENDING_WRITES = max(0, _PENDING_WRITES - 1)

    future = _EXECUTOR.submit(wrapped)
    with _LOCK:
        _FUTURES.add(future)
    future.add_done_callback(lambda completed: _discard_future(completed))
    return future


def _discard_future(future: Future[Any]) -> None:
    with _LOCK:
        _FUTURES.discard(future)


def record_session_start_async(
    *,
    session_id: str,
    task_id: str,
    seed: int | None,
    observation: Any,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> Future[Any] | None:
    return _schedule(
        _write_session_start,
        session_id=session_id,
        task_id=task_id,
        seed=seed,
        observation=observation,
        llm_provider=llm_provider,
        llm_model=llm_model,
    )


def record_step_async(
    *,
    session_id: str,
    task_id: str,
    step_index: int,
    observation: Any,
    action: Any,
    reward: Any,
    done: bool,
    grade: float | None,
    error: str | None = None,
) -> Future[Any] | None:
    return _schedule(
        _write_step,
        session_id=session_id,
        task_id=task_id,
        step_index=step_index,
        observation=observation,
        action=action,
        reward=reward,
        done=done,
        grade=grade,
        error=error,
    )


def record_episode_end_async(
    *,
    session_id: str,
    task_id: str,
    steps: int,
    score: float,
    success: bool,
    rewards: list[float],
    error: str | None = None,
) -> Future[Any] | None:
    return _schedule(
        _write_episode_end,
        session_id=session_id,
        task_id=task_id,
        steps=steps,
        score=score,
        success=success,
        rewards=rewards,
        error=error,
    )


def get_memory_status() -> dict[str, Any]:
    client = _get_client()
    with _LOCK:
        pending_writes = _PENDING_WRITES
        dropped_writes = _DROPPED_WRITES
    return {
        "memory_enabled": client is not None,
        "pending_writes": pending_writes,
        "dropped_writes": dropped_writes,
        "collection": _collection_name(),
        "last_error": _INIT_ERROR,
    }


def shutdown_memory_writer(timeout: float = 10.0) -> None:
    with _LOCK:
        futures = list(_FUTURES)
    if not futures:
        return
    wait(futures, timeout=timeout)
