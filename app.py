# =============================================================================
# File: app.py
# Owner: Aryan
# =============================================================================

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from nova_rl_env.config import load_env_file

load_env_file()

from nova_rl_env.environment import NovaRLEnv
from nova_rl_env.llm import (
    build_llm_client,
    fallback_action,
    format_llm_label,
    get_llm_action,
    get_supported_providers,
    is_fatal_llm_error,
    resolve_llm_config,
    resolve_llm_config_with_overrides,
)
from nova_rl_env.memory import (
    generate_session_id,
    get_memory_status,
    record_session_start_async,
    record_step_async,
)
from nova_rl_env.models import Action, LLMConfig, LLMProviderType, TaskId

app = FastAPI(title="Nova RL")

_DEFAULT_ALLOWED_ORIGINS = [
    "http://127.0.0.1:8080",
    "http://localhost:8080",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]
_allowed_origins = [
    origin.strip()
    for origin in os.getenv("NOVA_RL_CORS_ORIGINS", "").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or _DEFAULT_ALLOWED_ORIGINS,
    allow_origin_regex=None if _allowed_origins else r"https://.*\.(web\.app|firebaseapp\.com)$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

_SESSION_LOCK = Lock()


@dataclass
class SessionRuntime:
    env: NovaRLEnv
    touched_at: datetime
    llm_config: LLMConfig
    firebase_logging_enabled: bool
    firebase_credentials_path: str | None
    firebase_project: str | None


_SESSIONS: dict[str, SessionRuntime] = {}
_SESSION_TTL_SECONDS = 3600  # 1 hour


def _load_root_html() -> str:
    candidate_paths = [
        Path(__file__).with_name("ui").joinpath("public", "index.html"),
        Path(__file__).with_name("ui").joinpath("nova_ui.html"),
        Path(__file__).with_name("public").joinpath("index.html"),
        Path(__file__).with_name("nova_ui.html"),
    ]
    for path in candidate_paths:
        if path.exists():
            logger.info("Serving UI from %s", path)
            return path.read_text(encoding="utf-8")
    logger.warning("No UI HTML found; using fallback response")
    return "<html><body>Nova RL Server OK</body></html>"


ROOT_HTML = _load_root_html()


class ResetRequest(BaseModel):
    task_id: TaskId = "easy"
    seed: int | None = Field(default=None, ge=0, le=2147483647)
    session_id: str | None = None
    llm_provider: LLMProviderType | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_api_url: str | None = None
    firebase_logging_enabled: bool = False
    google_application_credentials: str | None = None
    google_cloud_project: str | None = None


def _serialize_llm_config(config: LLMConfig) -> dict[str, object]:
    return {
        "provider": config.provider,
        "model": config.model,
        "api_key_present": bool(config.api_key),
        "api_url": config.api_url,
    }


def _session_logging_enabled(session: SessionRuntime) -> bool:
    return session.firebase_logging_enabled


def _cleanup_expired_sessions() -> None:
    """Remove expired sessions to prevent memory leak."""
    now = datetime.now(timezone.utc)
    with _SESSION_LOCK:
        expired = [
            sid for sid, session in _SESSIONS.items()
            if (now - session.touched_at).total_seconds() > _SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del _SESSIONS[sid]
            logger.info(f"Cleaned up expired session: {sid}")


def _get_session_runtime(session_id: str) -> SessionRuntime:
    session = _SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown session_id: {session_id}")
    return session


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


@app.get("/providers")
def providers() -> dict[str, object]:
    return {
        "providers": get_supported_providers(),
        "default": resolve_llm_config().model_dump(),
    }


def _reset_session(
    task_id: TaskId = "easy",
    seed: int | None = None,
    session_id: str | None = None,
    llm_provider: LLMProviderType | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_api_url: str | None = None,
    firebase_logging_enabled: bool = False,
    google_application_credentials: str | None = None,
    google_cloud_project: str | None = None,
) -> dict[str, object]:
    resolved_session_id = session_id or generate_session_id(task_id)
    now = datetime.now(timezone.utc)
    llm_config = resolve_llm_config_with_overrides(
        llm_provider,
        llm_model,
        llm_api_key,
        llm_api_url,
    )
    with _SESSION_LOCK:
        session = _SESSIONS.get(resolved_session_id)
        if session is None:
            env = NovaRLEnv(task_id=task_id)
            _SESSIONS[resolved_session_id] = SessionRuntime(
                env=env,
                touched_at=now,
                llm_config=llm_config,
                firebase_logging_enabled=firebase_logging_enabled,
                firebase_credentials_path=(google_application_credentials or "").strip() or None,
                firebase_project=(google_cloud_project or "").strip() or None,
            )
        else:
            env = session.env
            _SESSIONS[resolved_session_id] = SessionRuntime(
                env=env,
                touched_at=now,
                llm_config=llm_config,
                firebase_logging_enabled=firebase_logging_enabled,
                firebase_credentials_path=(google_application_credentials or "").strip() or None,
                firebase_project=(google_cloud_project or "").strip() or None,
            )
        env.set_task(task_id)
        observation = env.reset(seed=seed)
        resolved_seed = env.seed
    logger.info("Reset session=%s task=%s llm=%s", resolved_session_id, task_id, format_llm_label(llm_config))
    if firebase_logging_enabled:
        record_session_start_async(
            session_id=resolved_session_id,
            task_id=task_id,
            seed=resolved_seed,
            observation=observation,
            llm_provider=llm_config.provider,
            llm_model=llm_config.model,
        )
    return {
        "status": "reset",
        "session_id": resolved_session_id,
        "task_id": task_id,
        "seed": resolved_seed,
        "llm_config": _serialize_llm_config(llm_config),
        "firebase_logging_enabled": firebase_logging_enabled,
        "observation": observation.model_dump(),
    }


@app.get("/reset")
def reset(
    task_id: TaskId = Query(default="easy"),
    seed: int | None = Query(default=None, ge=0),
    session_id: str | None = Query(default=None),
    llm_provider: LLMProviderType | None = Query(default=None),
    llm_model: str | None = Query(default=None),
    firebase_logging_enabled: bool = Query(default=False),
) -> dict[str, object]:
    return _reset_session(
        task_id=task_id,
        seed=seed,
        session_id=session_id,
        llm_provider=llm_provider,
        llm_model=llm_model,
        firebase_logging_enabled=firebase_logging_enabled,
    )


@app.post("/reset")
def reset_post(payload: ResetRequest | None = None) -> dict[str, object]:
    request = payload or ResetRequest()
    return _reset_session(
        task_id=request.task_id,
        seed=request.seed,
        session_id=request.session_id,
        llm_provider=request.llm_provider,
        llm_model=request.llm_model,
        llm_api_key=request.llm_api_key,
        llm_api_url=request.llm_api_url,
        firebase_logging_enabled=request.firebase_logging_enabled,
        google_application_credentials=request.google_application_credentials,
        google_cloud_project=request.google_cloud_project,
    )


@app.get("/state")
def state(session_id: str = Query(...)) -> dict[str, object]:
    _cleanup_expired_sessions()
    with _SESSION_LOCK:
        session = _get_session_runtime(session_id)
        env = session.env
        return {
            "session_id": session_id,
            "llm_config": _serialize_llm_config(session.llm_config),
            "firebase_logging_enabled": session.firebase_logging_enabled,
            "observation": env.state().model_dump(),
        }


@app.post("/step")
def step(action: Action, session_id: str = Query(...)) -> dict[str, object]:
    _cleanup_expired_sessions()
    try:
        with _SESSION_LOCK:
            session = _get_session_runtime(session_id)
            env = session.env
            observation, reward, done, info = env.step(action)
            task_id = env.task_id
            step_index = env.step_index
            _SESSIONS[session_id] = SessionRuntime(
                env=env,
                touched_at=datetime.now(timezone.utc),
                llm_config=session.llm_config,
                firebase_logging_enabled=session.firebase_logging_enabled,
                firebase_credentials_path=session.firebase_credentials_path,
                firebase_project=session.firebase_project,
            )
        grade = info.get("grade")
        if session.firebase_logging_enabled:
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
            "llm_config": _serialize_llm_config(session.llm_config),
            "firebase_logging_enabled": session.firebase_logging_enabled,
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


@app.post("/agent/step")
def agent_step(session_id: str = Query(...)) -> dict[str, object]:
    _cleanup_expired_sessions()
    error: str | None = None
    with _SESSION_LOCK:
        session = _get_session_runtime(session_id)
        env = session.env
        llm_config = session.llm_config
        obs = env.state()
        _SESSIONS[session_id] = SessionRuntime(
            env=env,
            touched_at=datetime.now(timezone.utc),
            llm_config=llm_config,
            firebase_logging_enabled=session.firebase_logging_enabled,
            firebase_credentials_path=session.firebase_credentials_path,
            firebase_project=session.firebase_project,
        )

    try:
        client = build_llm_client(llm_config)
        action = get_llm_action(client, llm_config, obs)
    except Exception as exc:
        error = str(exc).replace("\r", " ").replace("\n", " ").strip()
        if is_fatal_llm_error(error):
            logger.warning("Agent step fallback provider=%s model=%s error=%s", llm_config.provider, llm_config.model, error)
        action = fallback_action(obs, error=error)

    with _SESSION_LOCK:
        session = _get_session_runtime(session_id)
        env = session.env
        observation, reward, done, info = env.step(action)
        task_id = env.task_id
        step_index = env.step_index
        _SESSIONS[session_id] = SessionRuntime(
            env=env,
            touched_at=datetime.now(timezone.utc),
            llm_config=session.llm_config,
            firebase_logging_enabled=session.firebase_logging_enabled,
            firebase_credentials_path=session.firebase_credentials_path,
            firebase_project=session.firebase_project,
        )

    grade = info.get("grade")
    if session.firebase_logging_enabled:
        record_step_async(
            session_id=session_id,
            task_id=task_id,
            step_index=step_index,
            observation=observation,
            action=action,
            reward=reward,
            done=done,
            grade=float(grade) if isinstance(grade, (int, float)) else None,
            error=error,
        )
    return {
        "session_id": session_id,
        "llm_config": _serialize_llm_config(llm_config),
        "firebase_logging_enabled": session.firebase_logging_enabled,
        "action": action.model_dump(),
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
        "error": error,
    }


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return ROOT_HTML
