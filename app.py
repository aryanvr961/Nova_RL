# =============================================================================
# File: app.py
# Owner: Aryan
# =============================================================================

import logging
from pathlib import Path
from threading import Lock
from datetime import datetime, timedelta, timezone

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from nova_rl_env.config import load_env_file

load_env_file()

from nova_rl_env.environment import NovaRLEnv
from nova_rl_env.memory import (
    generate_session_id,
    get_memory_status,
    record_session_start_async,
    record_step_async,
)
from nova_rl_env.models import Action, TaskId

app = FastAPI(title="Nova RL")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SESSION_LOCK = Lock()
_SESSIONS: dict[str, tuple[NovaRLEnv, datetime]] = {}
_SESSION_TTL_SECONDS = 3600  # 1 hour

try:
    ROOT_HTML = Path(__file__).with_name("nova_ui.html").read_text(encoding="utf-8")
except FileNotFoundError:
    logger.warning("nova_ui.html not found; using fallback")
    ROOT_HTML = "<html><body>Nova RL Server OK</body></html>"


class ResetRequest(BaseModel):
    task_id: TaskId = "easy"
    seed: int | None = Field(default=None, ge=0, le=2147483647)
    session_id: str | None = None


def _cleanup_expired_sessions() -> None:
    """Remove expired sessions to prevent memory leak."""
    now = datetime.now(timezone.utc)
    expired = [
        sid for sid, (_, created_at) in _SESSIONS.items()
        if (now - created_at).total_seconds() > _SESSION_TTL_SECONDS
    ]
    for sid in expired:
        del _SESSIONS[sid]
        logger.info(f"Cleaned up expired session: {sid}")


def _get_session_env(session_id: str) -> NovaRLEnv:
    _cleanup_expired_sessions()
    item = _SESSIONS.get(session_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    env, _ = item
    return env


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, object]:
    memory_status = get_memory_status()
    return {
        "status": "healthy",
        "memory_enabled": memory_status["memory_enabled"],
        "pending_writes": memory_status["pending_writes"],
    }


def _reset_session(
    task_id: TaskId = "easy",
    seed: int | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    resolved_session_id = session_id or generate_session_id(task_id)
    now = datetime.now(timezone.utc)
    with _SESSION_LOCK:
        item = _SESSIONS.get(resolved_session_id)
        if item is None:
            env = NovaRLEnv(task_id=task_id)
            _SESSIONS[resolved_session_id] = (env, now)
        else:
            env, _ = item
            _SESSIONS[resolved_session_id] = (env, now)
        env.set_task(task_id)
        observation = env.reset(seed=seed)
        resolved_seed = env.seed
    logger.info(f"Reset session: {resolved_session_id} task={task_id}")
    record_session_start_async(
        session_id=resolved_session_id,
        task_id=task_id,
        seed=resolved_seed,
        observation=observation,
    )
    return {
        "status": "reset",
        "session_id": resolved_session_id,
        "task_id": task_id,
        "seed": resolved_seed,
        "observation": observation.model_dump(),
    }


@app.get("/reset")
def reset(
    task_id: TaskId = Query(default="easy"),
    seed: int | None = Query(default=None, ge=0),
    session_id: str | None = Query(default=None),
) -> dict[str, object]:
    return _reset_session(task_id=task_id, seed=seed, session_id=session_id)


@app.post("/reset")
def reset_post(payload: ResetRequest | None = None) -> dict[str, object]:
    request = payload or ResetRequest()
    return _reset_session(
        task_id=request.task_id,
        seed=request.seed,
        session_id=request.session_id,
    )


@app.get("/state")
def state(session_id: str = Query(...)) -> dict[str, object]:
    with _SESSION_LOCK:
        env = _get_session_env(session_id)
        return {
            "session_id": session_id,
            "observation": env.state().model_dump(),
        }


@app.post("/step")
def step(action: Action, session_id: str = Query(...)) -> dict[str, object]:
    # Validate action decision
    if action.decision not in ["fix", "quarantine", "promote", "noop", "finalize"]:
        logger.warning(f"Invalid decision: {action.decision}")
        raise HTTPException(status_code=400, detail=f"Invalid decision: {action.decision}")
    
    # Validate threshold range
    if not (0.0 <= action.threshold <= 1.0):
        logger.warning(f"Invalid threshold: {action.threshold}")
        raise HTTPException(status_code=400, detail="Threshold must be in [0.0, 1.0]")
    
    try:
        with _SESSION_LOCK:
            env = _get_session_env(session_id)
            observation, reward, done, info = env.step(action)
            task_id = env.task_id
            step_index = env.step_index
        grade = info.get("grade")
        record_step_async(
            session_id=session_id,
            task_id=task_id,
            step_index=step_index,
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            grade=float(grade) if isinstance(grade, (int, float)) else None,
        )
        logger.info(f"Step completed: session={session_id} step={step_index} decision={action.decision}")
        return {
            "session_id": session_id,
            "observation": observation.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Step failed for session {session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return ROOT_HTML
