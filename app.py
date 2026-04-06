# =============================================================================
# File: app.py
# Owner: Aryan
# Role: Hugging Face startup entrypoint
# =============================================================================

from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

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


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Nova RL</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #08111f;
      --panel: rgba(14, 24, 40, 0.92);
      --panel-alt: rgba(10, 18, 31, 0.88);
      --line: rgba(114, 176, 255, 0.16);
      --text: #edf4ff;
      --muted: #9bb0cc;
      --cyan: #62e6ff;
      --lime: #7ef7ba;
      --amber: #ffcb6b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(68, 139, 255, 0.26), transparent 34%),
        radial-gradient(circle at top right, rgba(126, 247, 186, 0.18), transparent 28%),
        linear-gradient(135deg, #06101c, #08111f 42%, #0b1627);
    }
    .shell {
      width: min(1080px, calc(100% - 32px));
      margin: 32px auto;
      padding: 28px;
      border: 1px solid var(--line);
      border-radius: 24px;
      background: var(--panel);
      box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
    }
    .hero {
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 18px;
      align-items: stretch;
    }
    .hero-card, .panel {
      border: 1px solid var(--line);
      border-radius: 20px;
      background: var(--panel-alt);
      padding: 20px;
    }
    .eyebrow {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(98, 230, 255, 0.08);
      color: var(--cyan);
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
    }
    h1 {
      margin: 16px 0 10px;
      font-size: clamp(34px, 6vw, 58px);
      line-height: 0.96;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.6;
    }
    .status {
      display: grid;
      gap: 12px;
    }
    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      width: fit-content;
      padding: 10px 14px;
      border-radius: 999px;
      color: #071019;
      background: linear-gradient(90deg, var(--lime), #d7ff8d);
      font-weight: 700;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: #0b1220;
    }
    .mini {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
      margin-top: 20px;
    }
    .mini div {
      padding: 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.02);
    }
    .mini strong {
      display: block;
      margin-bottom: 6px;
      color: var(--amber);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .section-title {
      margin: 28px 0 12px;
      font-size: 12px;
      color: var(--cyan);
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 16px;
    }
    .endpoint {
      padding: 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.02);
    }
    .method {
      display: inline-block;
      min-width: 52px;
      padding: 6px 10px;
      border-radius: 999px;
      background: rgba(126, 247, 186, 0.12);
      color: var(--lime);
      font-size: 12px;
      font-weight: 700;
      text-align: center;
      margin-bottom: 12px;
    }
    code {
      color: var(--text);
      font-family: Consolas, "Courier New", monospace;
      word-break: break-all;
    }
    a {
      color: var(--cyan);
      text-decoration: none;
    }
    a:hover { text-decoration: underline; }
    .footnote {
      margin-top: 18px;
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(98, 230, 255, 0.05);
      color: var(--muted);
    }
    @media (max-width: 820px) {
      .hero, .grid, .mini { grid-template-columns: 1fr; }
      .shell { width: min(100%, calc(100% - 20px)); margin: 10px auto; padding: 18px; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="hero-card">
        <span class="eyebrow">OpenEnv • ETL Remediation</span>
        <h1>Nova RL</h1>
        <p>
          A lightweight reinforcement-learning environment for ETL data cleanup.
          Agents observe noisy tabular batches, choose remediation actions, and
          receive deterministic rewards across easy, medium, and hard tasks.
        </p>
        <div class="mini">
          <div>
            <strong>Tasks</strong>
            easy, medium, hard
          </div>
          <div>
            <strong>Actions</strong>
            fix, quarantine, promote, noop, finalize
          </div>
          <div>
            <strong>Status</strong>
            ready for API evaluation
          </div>
        </div>
      </div>
      <div class="hero-card status">
        <span class="status-badge"><span class="dot"></span> Running</span>
        <p>
          This page is a minimal landing view for Hugging Face Spaces.
          Automated evaluation should use the API endpoints below.
        </p>
        <div class="footnote">
          Start a run with
          <code><a href="/reset?task_id=easy&seed=42&session_id=demo">/reset?task_id=easy&seed=42&session_id=demo</a></code>
          and continue with <code>/state</code> and <code>/step</code> using the same
          <code>session_id</code>.
        </div>
      </div>
    </section>

    <div class="section-title">API Surface</div>
    <section class="grid">
      <article class="endpoint">
        <div class="method">GET</div>
        <div><code>/ping</code></div>
        <p>Liveness check for automated validators.</p>
      </article>
      <article class="endpoint">
        <div class="method">GET</div>
        <div><code>/health</code></div>
        <p>Runtime health signal for the deployed service.</p>
      </article>
      <article class="endpoint">
        <div class="method">GET</div>
        <div><code>/reset?task_id=easy&seed=42&session_id=demo</code></div>
        <p>Initializes or resets a task episode and returns the first observation.</p>
      </article>
      <article class="endpoint">
        <div class="method">GET</div>
        <div><code>/state?session_id=demo</code></div>
        <p>Returns the current observation for an existing session.</p>
      </article>
      <article class="endpoint">
        <div class="method">POST</div>
        <div><code>/step?session_id=demo</code></div>
        <p>Accepts an action payload and returns observation, reward, and grading info.</p>
      </article>
      <article class="endpoint">
        <div class="method">JSON</div>
        <div><code>{"decision":"fix","threshold":0.6,"notes":"smoke"}</code></div>
        <p>Example request body for the <code>/step</code> endpoint.</p>
      </article>
    </section>
  </main>
</body>
</html>
"""
