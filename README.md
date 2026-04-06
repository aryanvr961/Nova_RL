---
title: Nova RL
emoji: 🚀
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Nova RL

Nova RL is an OpenEnv environment for ETL data remediation. The agent receives noisy tabular batches, takes structured remediation actions, and is scored with deterministic graders across `easy`, `medium`, and `hard` tasks.

## Environment Summary

- Domain: ETL data quality remediation
- Interface: typed `reset()`, `step(action)`, and `state()`
- Tasks: `easy`, `medium`, `hard`
- Score range: deterministic `0.0` to `1.0`

## Real-World Task

The benchmark simulates common ETL cleanup failures:

- missing values
- duplicate rows
- malformed dates
- type mismatches
- schema drift

## Observation Space

The observation payload exposes:

- `task_id`
- `step_index`
- `max_steps`
- `batch_size`
- `anomaly_counts`
- `current_metrics`
- `sample_issue_summaries`
- `current_threshold`
- `remaining_steps`
- `last_action`

## Action Space

The agent can return:

- `fix`
- `quarantine`
- `promote`
- `noop`
- `finalize`

Each action includes a bounded `threshold`, optional `notes`, and optional structured `parameters`.

## Task Definitions

- Easy: handle null values and exact duplicate rows safely
- Medium: handle type mismatches and malformed dates
- Hard: handle schema drift and correlated multi-column issues

Task configs live in `nova_rl_env/tasks.py`, synthetic batch generation lives in `nova_rl_env/datagen.py`, and deterministic grading lives in `nova_rl_env/graders.py`.

## Repository Layout

```text
NOVA_RL/
|-- app.py
|-- Dockerfile
|-- inference.py
|-- openenv.yaml
|-- README.md
|-- requirements.txt
`-- nova_rl_env/
    |-- datagen.py
    |-- environment.py
    |-- graders.py
    |-- models.py
    |-- rewards.py
    `-- tasks.py
```

## Setup

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and define the inference variables before running `inference.py`.

## Runtime API

The FastAPI app exposes:

- `GET /ping`
- `GET /health`
- `GET /reset`
- `GET /state`
- `POST /step`

`/reset` creates or resets a session-scoped environment and returns a `session_id`. `/state` and `/step` require that `session_id`.

## Inference Configuration

The baseline inference script uses the OpenAI Python client with environment variables aligned to the requirement doc:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Compatibility fallbacks also supported:

- `OPENAI_API_KEY`
- `API_KEY`

By default the script targets the Hugging Face router endpoint and reads `MODEL_NAME` from the environment.

## Deployment

The project is packaged for Hugging Face Space deployment:

- `Dockerfile`
- `openenv.yaml`
- `app.py`

`openenv.yaml` uses an OpenEnv-style FastAPI Space manifest with `spec_version`, `runtime`, `app`, and `port`.

## Baseline Scores

Representative task/grader integration checks produced valid scores in the required range:

- Easy: `0.8`
- Medium: `0.5375`
- Hard: `0.4725`

## Ownership

- Aryan: environment, models, rewards, inference, deployment-facing files
- Aadyaa: tasks, data generation, grading logic, and task-facing documentation

<!-- GitHub-friendly pure markdown format. All CSS styling removed for proper rendering on GitHub. -->

.nova-shell {
  position: relative;
  margin: 24px auto;
  max-width: 1040px;
  padding: 28px;
  color: var(--text);
  background:
    radial-gradient(circle at top left, rgba(0, 255, 136, 0.10), transparent 34%),
    radial-gradient(circle at bottom right, rgba(0, 217, 255, 0.08), transparent 30%),
    linear-gradient(180deg, #07120d 0%, #050b08 100%);
  border: 1px solid var(--line);
  border-radius: 24px;
  overflow: hidden;
  box-shadow:
    0 0 0 1px rgba(0, 255, 136, 0.07) inset,
    0 24px 80px rgba(0, 0, 0, 0.45),
    0 0 60px rgba(0, 255, 136, 0.08);
  font-family: "Segoe UI", Arial, sans-serif;
}

.nova-shell::before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(rgba(0, 255, 136, 0.035) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0, 255, 136, 0.035) 1px, transparent 1px);
  background-size: 26px 26px;
  pointer-events: none;
  opacity: 0.6;
}

.nova-shell::after {
  content: "";
  position: absolute;
  inset: 0;
  background: repeating-linear-gradient(180deg, transparent 0, transparent 3px, rgba(255, 255, 255, 0.015) 4px);
  mix-blend-mode: screen;
  pointer-events: none;
}

.nova-orb {
  position: absolute;
  border-radius: 999px;
  filter: blur(48px);
  opacity: 0.18;
  pointer-events: none;
  animation: floatOrb 12s ease-in-out infinite;
}

.orb-a {
  top: -90px;
  left: -70px;
  width: 260px;
  height: 260px;
  background: rgba(0, 255, 136, 0.45);
}

.orb-b {
  right: -60px;
  bottom: -80px;
  width: 220px;
  height: 220px;
  background: rgba(0, 217, 255, 0.28);
  animation-delay: -4s;
}

.nova-content {
  position: relative;
  z-index: 1;
}

.hero {
  position: relative;
  text-align: center;
  padding: 20px 16px 10px;
  animation: fadeUp 0.8s ease both;
}

.hero-glow {
  position: absolute;
  top: 20px;
  left: 50%;
  transform: translateX(-50%);
  width: 70%;
  height: 180px;
  background: radial-gradient(circle, rgba(0, 255, 136, 0.16), transparent 68%);
  filter: blur(8px);
  animation: pulseGlow 4s ease-in-out infinite;
  pointer-events: none;
}

.badge-row {
  display: flex;
  justify-content: center;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 18px;
}

.badge {
  padding: 7px 12px;
  border-radius: 999px;
  border: 1px solid var(--line);
  background: rgba(10, 24, 18, 0.78);
  font-size: 11px;
  letter-spacing: 1.4px;
  text-transform: uppercase;
  color: var(--muted);
  font-family: Consolas, "Courier New", monospace;
  animation: chipFloat 3s ease-in-out infinite;
}

.badge:nth-child(2) { animation-delay: 0.3s; }
.badge:nth-child(3) { animation-delay: 0.6s; }
.badge:nth-child(4) { animation-delay: 0.9s; }
.badge:nth-child(5) { animation-delay: 1.2s; }
.badge:nth-child(6) { animation-delay: 1.5s; }

.badge.green { color: var(--neon); border-color: rgba(0, 255, 136, 0.35); }
.badge.blue { color: var(--cyan); border-color: rgba(0, 217, 255, 0.35); }
.badge.yellow { color: var(--yellow); border-color: rgba(255, 213, 74, 0.35); }
.badge.red { color: var(--red); border-color: rgba(255, 92, 122, 0.35); }

.hero-title {
  position: relative;
  margin: 0;
  font-size: clamp(40px, 7vw, 74px);
  line-height: 1;
  letter-spacing: 8px;
  color: var(--neon);
  text-shadow: 0 0 12px rgba(0, 255, 136, 0.55), 0 0 34px rgba(0, 255, 136, 0.20);
  animation: flicker 7s linear infinite, heroLift 5s ease-in-out infinite;
}

.hero-sub {
  margin-top: 12px;
  color: var(--muted);
  font-family: Consolas, "Courier New", monospace;
  letter-spacing: 3px;
  font-size: 13px;
  text-transform: uppercase;
}

.hero-desc {
  max-width: 720px;
  margin: 18px auto 0;
  font-size: 15px;
  line-height: 1.8;
  color: #bde8cf;
}

.divider {
  display: flex;
  align-items: center;
  gap: 12px;
  margin: 28px 0 18px;
  color: var(--neon);
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
  letter-spacing: 2px;
  text-transform: uppercase;
}

.divider::before,
.divider::after {
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--line-strong), transparent);
}

.section-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
  color: var(--neon);
  letter-spacing: 1px;
  text-transform: uppercase;
  font-size: 18px;
}

.section-title span {
  font-size: 20px;
  animation: chipFloat 2.8s ease-in-out infinite;
}

.grid-4,
.grid-5,
.grid-3,
.grid-2 {
  display: grid;
  gap: 14px;
}

.grid-4 { grid-template-columns: repeat(4, minmax(0, 1fr)); }
.grid-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.grid-3 { grid-template-columns: repeat(3, minmax(0, 1fr)); }
.grid-2 { grid-template-columns: repeat(2, minmax(0, 1fr)); }

.card,
.stat-card,
.reward-card,
.purpose-card,
.panel {
  position: relative;
  padding: 18px;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(15, 33, 25, 0.95), rgba(8, 18, 13, 0.96));
  border: 1px solid rgba(0, 255, 136, 0.18);
  box-shadow: 0 8px 26px rgba(0, 0, 0, 0.24);
  overflow: hidden;
}

.card::before,
.stat-card::before,
.reward-card::before,
.purpose-card::before,
.panel::before {
  content: "";
  position: absolute;
  inset: 0 auto 0 0;
  width: 3px;
  background: linear-gradient(180deg, var(--neon), transparent);
}

.stat-card,
.reward-card,
.purpose-card {
  animation: cardFloat 4.2s ease-in-out infinite;
}

.stat-card:nth-child(2),
.reward-card:nth-child(2),
.purpose-card:nth-child(2) { animation-delay: 0.3s; }
.stat-card:nth-child(3),
.reward-card:nth-child(3),
.purpose-card:nth-child(3) { animation-delay: 0.6s; }
.stat-card:nth-child(4),
.reward-card:nth-child(4),
.purpose-card:nth-child(4) { animation-delay: 0.9s; }
.reward-card:nth-child(5),
.purpose-card:nth-child(5) { animation-delay: 1.2s; }
.purpose-card:nth-child(6) { animation-delay: 1.5s; }

.stat-num {
  display: block;
  font-size: 28px;
  font-weight: 700;
  color: var(--neon);
  text-shadow: 0 0 10px rgba(0, 255, 136, 0.34);
}

.stat-label {
  display: block;
  margin-top: 6px;
  font-size: 11px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 1.2px;
  font-family: Consolas, "Courier New", monospace;
}

.scoreboard {
  border-radius: 18px;
  overflow: hidden;
  border: 1px solid rgba(0, 255, 136, 0.18);
  background: rgba(9, 19, 14, 0.96);
}

.score-head,
.score-row {
  display: grid;
  grid-template-columns: 140px 1fr 110px 90px;
  gap: 12px;
  align-items: center;
  padding: 14px 18px;
}

.score-head {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.6px;
  font-family: Consolas, "Courier New", monospace;
  background: linear-gradient(90deg, rgba(0, 255, 136, 0.10), transparent);
  border-bottom: 1px solid rgba(0, 255, 136, 0.14);
}

.score-row + .score-row {
  border-top: 1px solid rgba(0, 255, 136, 0.08);
}

.pill {
  display: inline-block;
  padding: 6px 12px;
  border-radius: 999px;
  font-size: 11px;
  letter-spacing: 1px;
  font-family: Consolas, "Courier New", monospace;
  text-transform: uppercase;
  border: 1px solid;
}

.easy { color: #72ff7b; border-color: rgba(114, 255, 123, 0.3); background: rgba(114, 255, 123, 0.10); }
.medium { color: var(--yellow); border-color: rgba(255, 213, 74, 0.3); background: rgba(255, 213, 74, 0.10); }
.hard { color: var(--red); border-color: rgba(255, 92, 122, 0.3); background: rgba(255, 92, 122, 0.10); }

.bar {
  height: 10px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.05);
  overflow: hidden;
  position: relative;
}

.bar > span {
  display: block;
  height: 100%;
  border-radius: inherit;
  transform-origin: left;
  animation: growBar 2.2s ease forwards;
  box-shadow: 0 0 16px currentColor;
}

.easy-bar > span { width: 80%; color: #72ff7b; background: linear-gradient(90deg, #1b8f38, #72ff7b); }
.medium-bar > span { width: 53.75%; color: var(--yellow); background: linear-gradient(90deg, #a67d12, #ffd54a); }
.hard-bar > span { width: 47.25%; color: var(--red); background: linear-gradient(90deg, #9e2742, #ff5c7a); }

.score-val {
  text-align: right;
  font-weight: 700;
}

.score-rank {
  text-align: right;
  color: var(--muted);
  font-size: 11px;
  font-family: Consolas, "Courier New", monospace;
}

table.nova-table {
  width: 100%;
  border-collapse: collapse;
  overflow: hidden;
  border-radius: 18px;
  background: rgba(9, 19, 14, 0.96);
  border: 1px solid rgba(0, 255, 136, 0.18);
}

.nova-table th,
.nova-table td {
  padding: 14px 12px;
  text-align: left;
  vertical-align: top;
  border-bottom: 1px solid rgba(0, 255, 136, 0.08);
}

.nova-table th {
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 1.5px;
  font-family: Consolas, "Courier New", monospace;
  background: rgba(0, 255, 136, 0.04);
}

.nova-table tr:last-child td {
  border-bottom: 0;
}

.tag {
  display: inline-block;
  margin: 2px 5px 2px 0;
  padding: 4px 8px;
  border-radius: 999px;
  font-size: 11px;
  font-family: Consolas, "Courier New", monospace;
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.tag.issue { color: var(--red); background: rgba(255, 92, 122, 0.08); border-color: rgba(255, 92, 122, 0.22); }
.tag.action { color: var(--cyan); background: rgba(0, 217, 255, 0.08); border-color: rgba(0, 217, 255, 0.22); }
.tag.reward { color: var(--violet); background: rgba(179, 136, 255, 0.08); border-color: rgba(179, 136, 255, 0.22); }

.reward-icon,
.purpose-icon {
  display: block;
  font-size: 26px;
  margin-bottom: 10px;
}

.reward-name,
.purpose-name {
  display: block;
  font-size: 12px;
  color: var(--neon);
  text-transform: uppercase;
  letter-spacing: 1.4px;
  font-family: Consolas, "Courier New", monospace;
  margin-bottom: 6px;
}

.reward-value {
  display: block;
  margin-bottom: 6px;
  font-size: 18px;
  font-weight: 700;
}

.muted {
  color: var(--muted);
}

.flow {
  display: flex;
  align-items: center;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 10px;
}

.node {
  min-width: 112px;
  padding: 14px 12px;
  text-align: center;
  border-radius: 16px;
  background: linear-gradient(180deg, rgba(14, 30, 23, 0.98), rgba(9, 18, 14, 0.98));
  border: 1px solid rgba(0, 255, 136, 0.18);
  animation: glowShift 4.5s ease-in-out infinite;
}

.node strong {
  display: block;
  color: var(--neon);
  font-size: 12px;
  font-family: Consolas, "Courier New", monospace;
  margin-bottom: 4px;
}

.node span {
  color: var(--muted);
  font-size: 11px;
}

.arrow {
  min-width: 34px;
  text-align: center;
  color: var(--neon);
  font-size: 18px;
  animation: arrowPulse 1.8s ease-in-out infinite;
}

.list-card ul {
  margin: 0;
  padding-left: 18px;
  line-height: 1.9;
}

.endpoint {
  display: grid;
  grid-template-columns: 84px 140px 1fr;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 14px;
  background: rgba(11, 24, 18, 0.92);
  border: 1px solid rgba(0, 255, 136, 0.14);
}

.endpoint + .endpoint {
  margin-top: 10px;
}

.method {
  display: inline-block;
  text-align: center;
  padding: 6px 10px;
  border-radius: 999px;
  font-family: Consolas, "Courier New", monospace;
  font-size: 11px;
  letter-spacing: 1px;
  border: 1px solid;
}

.method.get { color: var(--cyan); border-color: rgba(0, 217, 255, 0.25); background: rgba(0, 217, 255, 0.08); }
.method.post { color: var(--yellow); border-color: rgba(255, 213, 74, 0.25); background: rgba(255, 213, 74, 0.08); }

.path {
  color: var(--neon);
  font-family: Consolas, "Courier New", monospace;
  font-size: 13px;
}

pre.repo-tree {
  margin: 0;
  padding: 18px;
  border-radius: 18px;
  background: rgba(9, 19, 14, 0.96);
  border: 1px solid rgba(0, 255, 136, 0.18);
  color: #d7ffe9;
  overflow-x: auto;
  line-height: 1.7;
}

code {
  padding: 2px 6px;
  border-radius: 6px;
  background: rgba(0, 255, 136, 0.08);
  color: var(--neon);
  font-family: Consolas, "Courier New", monospace;
}

.footer {
  margin-top: 24px;
  padding-top: 16px;
  border-top: 1px solid rgba(0, 255, 136, 0.12);
  text-align: center;
}

.footer-logo {
  color: var(--neon);
  font-size: 28px;
  font-weight: 700;
  letter-spacing: 6px;
  text-shadow: 0 0 14px rgba(0, 255, 136, 0.25);
  margin-bottom: 10px;
}

.team {
  color: var(--muted);
  line-height: 1.9;
}

.footer-note {
  margin-top: 10px;
  color: var(--muted);
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
  letter-spacing: 1.2px;
}

@media (max-width: 920px) {
  .grid-5 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .grid-4 { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .grid-3 { grid-template-columns: 1fr; }
  .grid-2 { grid-template-columns: 1fr; }
  .score-head, .score-row { grid-template-columns: 1fr; }
  .endpoint { grid-template-columns: 1fr; }
}

@media (max-width: 640px) {
  .nova-shell { padding: 18px; border-radius: 18px; }
  .hero-title { letter-spacing: 4px; }
  .badge-row { justify-content: flex-start; }
  .grid-4, .grid-5 { grid-template-columns: 1fr; }
}

@keyframes pulseGlow {
  0%, 100% { opacity: 0.7; transform: translateX(-50%) scale(1); }
  50% { opacity: 1; transform: translateX(-50%) scale(1.05); }
}

@keyframes chipFloat {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-4px); }
}

@keyframes heroLift {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-5px); }
}

@keyframes flicker {
  0%, 18%, 22%, 62%, 64%, 100% { opacity: 1; }
  20%, 63% { opacity: 0.82; }
}

@keyframes growBar {
  from { transform: scaleX(0); }
  to { transform: scaleX(1); }
}

@keyframes arrowPulse {
  0%, 100% { opacity: 0.45; transform: translateX(0); }
  50% { opacity: 1; transform: translateX(3px); }
}

@keyframes glowShift {
  0%, 100% { box-shadow: 0 0 0 rgba(0, 255, 136, 0); }
  50% { box-shadow: 0 0 20px rgba(0, 255, 136, 0.12); }
}

@keyframes cardFloat {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-6px); }
}

@keyframes floatOrb {
  0%, 100% { transform: translate(0, 0) scale(1); }
  50% { transform: translate(18px, -16px) scale(1.05); }
}

@keyframes fadeUp {
  from { opacity: 0; transform: translateY(18px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>

<div class="nova-shell">
<div class="nova-orb orb-a"></div>
<div class="nova-orb orb-b"></div>
<div class="nova-content">
<div class="hero">
<div class="hero-glow"></div>
<div class="badge-row">
<span class="badge green">v1.0.0</span>
<span class="badge green">OpenEnv</span>
<span class="badge yellow">MIT License</span>
<span class="badge blue">Python 3.10+</span>
<span class="badge blue">FastAPI</span>
<span class="badge red">HuggingFace</span>
</div>
<div class="hero-title">NOVA RL</div>
<div class="hero-sub">ETL REMEDIATION BENCHMARK | REINFORCEMENT LEARNING</div>
<p class="hero-desc">A deterministic, score-driven benchmark where agents learn to repair noisy ETL pipelines across easy, medium, and hard episodes with real-world data quality stakes.</p>
</div>

<div class="divider">System Snapshot</div>
<div class="grid-4">
<div class="stat-card"><span class="stat-num">3</span><span class="stat-label">Difficulty Tiers</span></div>
<div class="stat-card"><span class="stat-num">5</span><span class="stat-label">Action Types</span></div>
<div class="stat-card"><span class="stat-num">0-1</span><span class="stat-label">Score Range</span></div>
<div class="stat-card"><span class="stat-num">REST</span><span class="stat-label">API Interface</span></div>
</div>

<div class="divider">Baseline Scoreboard</div>
<div class="scoreboard">
<div class="score-head">
<div>Difficulty</div>
<div>Progress</div>
<div style="text-align:right;">Score</div>
<div style="text-align:right;">Rank</div>
</div>
<div class="score-row">
<div><span class="pill easy">Easy</span></div>
<div class="bar easy-bar"><span></span></div>
<div class="score-val" style="color:#72ff7b;">0.8000</div>
<div class="score-rank">S Tier</div>
</div>
<div class="score-row">
<div><span class="pill medium">Medium</span></div>
<div class="bar medium-bar"><span></span></div>
<div class="score-val" style="color:var(--yellow);">0.5375</div>
<div class="score-rank">B Tier</div>
</div>
<div class="score-row">
<div><span class="pill hard">Hard</span></div>
<div class="bar hard-bar"><span></span></div>
<div class="score-val" style="color:var(--red);">0.4725</div>
<div class="score-rank">C Tier</div>
</div>
</div>

<div class="divider">Reward System</div>
<div class="section-title"><span>[R]</span> Action Reward Map</div>
<div class="grid-5">
<div class="reward-card"><span class="reward-icon">FIX</span><span class="reward-name">Fix</span><span class="reward-value" style="color:var(--neon);">+1.0</span><div class="muted">Correct repair of a valid anomaly</div></div>
<div class="reward-card"><span class="reward-icon">ISO</span><span class="reward-name">Quarantine</span><span class="reward-value" style="color:var(--yellow);">+0.5</span><div class="muted">Safe isolation of uncertain rows</div></div>
<div class="reward-card"><span class="reward-icon">GO</span><span class="reward-name">Promote</span><span class="reward-value" style="color:var(--cyan);">+0.8</span><div class="muted">Advancing clean records downstream</div></div>
<div class="reward-card"><span class="reward-icon">IDLE</span><span class="reward-name">Noop</span><span class="reward-value" style="color:var(--muted);">0.0</span><div class="muted">No-op on an anomaly wastes a step</div></div>
<div class="reward-card"><span class="reward-icon">END</span><span class="reward-name">Finalize</span><span class="reward-value" style="color:var(--violet);">+Bonus</span><div class="muted">Episode completion bonus if above threshold</div></div>
</div>

<div class="divider">Task Definitions</div>
<div class="section-title"><span>[T]</span> Episode Task Matrix</div>
<table class="nova-table">
<thead>
<tr>
<th>Level</th>
<th>Issues Handled</th>
<th>Actions</th>
<th>Reward Ceiling</th>
<th>Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td><span class="pill easy">Easy</span></td>
<td><span class="tag issue">null values</span><span class="tag issue">exact dupes</span></td>
<td><span class="tag action">fix</span><span class="tag action">promote</span></td>
<td><span class="tag reward">1.0</span></td>
<td class="muted">Deterministic patterns, high signal</td>
</tr>
<tr>
<td><span class="pill medium">Medium</span></td>
<td><span class="tag issue">type mismatch</span><span class="tag issue">bad dates</span></td>
<td><span class="tag action">fix</span><span class="tag action">quarantine</span><span class="tag action">promote</span></td>
<td><span class="tag reward">0.85</span></td>
<td class="muted">Ambiguous fields, repair or isolate</td>
</tr>
<tr>
<td><span class="pill hard">Hard</span></td>
<td><span class="tag issue">schema drift</span><span class="tag issue">multi-col corr.</span></td>
<td><span class="tag action">fix</span><span class="tag action">quarantine</span><span class="tag action">promote</span><span class="tag action">finalize</span></td>
<td><span class="tag reward">0.70</span></td>
<td class="muted">Cross-column causality, structural repair</td>
</tr>
</tbody>
</table>

<div class="divider">Execution Flow</div>
<div class="section-title"><span>[F]</span> Episode Execution Flow</div>
<div class="panel">
<div class="flow">
<div class="node"><strong>reset()</strong><span>new episode</span></div>
<div class="arrow">-&gt;</div>
<div class="node"><strong>state()</strong><span>observe</span></div>
<div class="arrow">-&gt;</div>
<div class="node"><strong>agent</strong><span>decide</span></div>
<div class="arrow">-&gt;</div>
<div class="node"><strong>step()</strong><span>execute</span></div>
<div class="arrow">-&gt;</div>
<div class="node"><strong>grade()</strong><span>score</span></div>
<div class="arrow">-&gt;</div>
<div class="node"><strong>end</strong><span>episode done</span></div>
</div>
<p class="muted" style="margin-top:12px;">Loop back to <code>state()</code> while more steps remain.</p>
</div>

<div class="divider">Real-World Purpose</div>
<div class="section-title"><span>[P]</span> Why This Benchmark Exists</div>
<div class="grid-3">
<div class="purpose-card"><span class="purpose-icon">01</span><span class="purpose-name">Data Pipelines</span>Financial and enterprise ETL pipelines lose millions to silent data corruption. NOVA RL trains agents that detect and repair these faults before they propagate.</div>
<div class="purpose-card"><span class="purpose-icon">02</span><span class="purpose-name">RL Research</span>A clean deterministic environment with discrete action spaces, useful for testing algorithms like PPO, SAC, or GRPO on structured tasks.</div>
<div class="purpose-card"><span class="purpose-icon">03</span><span class="purpose-name">Healthcare Data</span>Malformed clinical records directly affect patient safety. NOVA RL mirrors schema drift and type mismatch issues found in hospitals.</div>
<div class="purpose-card"><span class="purpose-icon">04</span><span class="purpose-name">Supply Chain</span>Duplicate and null-heavy inventory records create stock mismanagement. Easy-tier tasks simulate these frequent high-cost failures.</div>
<div class="purpose-card"><span class="purpose-icon">05</span><span class="purpose-name">ML Data Quality</span>Poor training data degrades model performance. Agents trained on NOVA RL can act as gatekeepers before features reach ML pipelines.</div>
<div class="purpose-card"><span class="purpose-icon">06</span><span class="purpose-name">LLM Eval</span>Evaluates instruction-following systems on structured decision-making, bridging text generation and tool-using agent workflows.</div>
</div>

<div class="divider">Observation and Action Space</div>
<div class="grid-2">
<div class="card list-card">
<div class="section-title"><span>[O]</span> Observation Space</div>
<ul>
<li><code>task_id</code></li>
<li><code>step_index</code></li>
<li><code>max_steps</code></li>
<li><code>batch_size</code></li>
<li><code>anomaly_counts</code></li>
<li><code>current_metrics</code></li>
<li><code>sample_issue_summaries</code></li>
<li><code>current_threshold</code></li>
<li><code>remaining_steps</code></li>
<li><code>last_action</code></li>
</ul>
</div>
<div class="card list-card">
<div class="section-title"><span>[A]</span> Action Space</div>
<ul>
<li><code>fix</code> repair the anomaly directly</li>
<li><code>quarantine</code> isolate for review</li>
<li><code>promote</code> advance clean records downstream</li>
<li><code>noop</code> skip and lose efficiency</li>
<li><code>finalize</code> commit and end episode</li>
</ul>
<p class="muted" style="margin-top:12px;">Each action accepts <code>threshold</code>, <code>notes</code>, and <code>params{}</code>.</p>
</div>
</div>

<div class="divider">API Endpoints</div>
<div class="section-title"><span>[API]</span> Runtime API</div>
<div class="panel">
<div class="endpoint"><div><span class="method get">GET</span></div><div class="path">/ping</div><div class="muted">Health check and service liveness confirmation.</div></div>
<div class="endpoint"><div><span class="method get">GET</span></div><div class="path">/health</div><div class="muted">Detailed service status and configuration view.</div></div>
<div class="endpoint"><div><span class="method get">GET</span></div><div class="path">/reset</div><div class="muted">Create or reset a session and return <code>session_id</code>.</div></div>
<div class="endpoint"><div><span class="method get">GET</span></div><div class="path">/state</div><div class="muted">Read the current observation for a given <code>session_id</code>.</div></div>
<div class="endpoint"><div><span class="method post">POST</span></div><div class="path">/step</div><div class="muted">Apply a remediation action and receive reward plus next state.</div></div>
</div>

<div class="divider">Repo Structure</div>
<div class="section-title"><span>[FS]</span> Repository Layout</div>
<pre class="repo-tree">NOVA_RL/
|-- app.py                 -> FastAPI entrypoint
|-- inference.py           -> baseline agent script
|-- preload_models.py       -> model preloading utility
|-- Dockerfile
|-- openenv.yaml           -> OpenEnv manifest
|-- requirements.txt
|-- README.md
`-- nova_rl_env/
    |-- __init__.py         -> package initialization
    |-- environment.py     -> reset / step / state logic
    |-- tasks.py           -> easy / medium / hard configs
    |-- graders.py         -> deterministic scoring
    |-- datagen.py         -> synthetic ETL batch generator
    |-- models.py
    `-- rewards.py         -> reward computation</pre>

<div class="footer">
<div class="footer-logo">NOVA RL</div>
<div class="team">Aryan -> environment, models, rewards, inference, deployment<br/>Aadyaa -> tasks, datagen, grading, documentation</div>
<div class="footer-note">MIT LICENSE | OPENENV COMPATIBLE | HUGGINGFACE SPACES READY</div>
</div>
</div>
</div>
