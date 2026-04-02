# =============================================================================
# File: app.py
# Owner: Aryan
# Role: Hugging Face startup entrypoint
# =============================================================================

from fastapi import FastAPI

from nova_rl_env.environment import NovaRLEnv

env = NovaRLEnv()
app = FastAPI(title="Nova RL")


@app.get("/ping")
def ping() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "Nova RL", "status": "ready"}
