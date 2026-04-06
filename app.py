# =============================================================================
# File: app.py
# Owner: Aryan
# Role: Hugging Face startup entrypoint
# =============================================================================

from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query

from nova_rl_env.environment import NovaRLEnv
from nova_rl_env.models import Action, TaskId

app = FastAPI(title="Nova RL")
_SESSION_LOCK = Lock()
_SESSIONS: dict[str, NovaRLEnv] = {}


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


@app.get("/reset")
def reset(
    task_id: TaskId = Query(default="easy"),
    seed: int | None = Query(default=None, ge=0),
    session_id: str | None = Query(default=None),
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


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "Nova RL", "status": "ready"}
