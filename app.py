# =============================================================================
# File: app.py
# Owner: Aryan
# Role: Hugging Face startup entrypoint
# =============================================================================

from pathlib import Path
from threading import Lock
from uuid import uuid4

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

from nova_rl_env.environment import NovaRLEnv
from nova_rl_env.models import Action, TaskId

app = FastAPI(title="Nova RL")
_SESSION_LOCK = Lock()
_SESSIONS: dict[str, NovaRLEnv] = {}
ROOT_HTML = Path(__file__).with_name("nova_ui.html").read_text(encoding="utf-8")


class ResetRequest(BaseModel):
    task_id: TaskId = "easy"
    seed: int | None = Field(default=None, ge=0)
    session_id: str | None = None


def _get_session_env(session_id: str) -> NovaRLEnv:
    env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return env


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


def _reset_session(
    task_id: TaskId = "easy",
    seed: int | None = None,
    session_id: str | None = None,
) -> dict[str, object]:
    resolved_session_id = session_id or str(uuid4())
    with _SESSION_LOCK:
        env = _SESSIONS.get(resolved_session_id)
        if env is None:
            env = NovaRLEnv(task_id=task_id)
            _SESSIONS[resolved_session_id] = env
        env.set_task(task_id)
        observation = env.reset(seed=seed)
    return {
        "status": "reset",
        "session_id": resolved_session_id,
        "task_id": task_id,
        "seed": env.seed,
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
    with _SESSION_LOCK:
        env = _get_session_env(session_id)
        observation, reward, done, info = env.step(action)
    return {
        "session_id": session_id,
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return ROOT_HTML
