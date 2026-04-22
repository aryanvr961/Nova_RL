from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Any

from google import genai
from google.genai import types

from .config import load_env_file
from .memory import truncate_text
from .models import Action, LLMConfig, Observation


load_env_file()

GEMINI_TIMEOUT_SECONDS = 15.0
GEMINI_MAX_RETRIES = 1
LOCAL_TIMEOUT_SECONDS = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "30"))
LOCAL_MAX_RETRIES = 1
LOCAL_GEMMA4_API_URL = os.getenv(
    "LOCAL_GEMMA4_API_URL",
    "http://127.0.0.1:1234/v1/chat/completions",
)
FATAL_ERROR_MARKERS = (
    "RESOURCE_EXHAUSTED",
    "UNAVAILABLE",
    "NOT_FOUND",
    "CONNECTION REFUSED",
    "FAILED TO ESTABLISH A NEW CONNECTION",
    "503",
    "429",
    "404",
)


def get_supported_providers() -> list[dict[str, str]]:
    return [
        {
            "provider": "gemini",
            "label": "Google Gemini",
            "default_model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        },
        {
            "provider": "local_gemma4",
            "label": "Local Gemma 4",
            "default_model": os.getenv("LOCAL_GEMMA4_MODEL", "gemma-4"),
        },
    ]


def resolve_llm_config(provider: str | None = None, model: str | None = None) -> LLMConfig:
    resolved_provider = (provider or os.getenv("LLM_PROVIDER", "gemini")).strip().lower()
    if resolved_provider not in {"gemini", "local_gemma4"}:
        raise ValueError(f"Unsupported llm provider: {resolved_provider}")

    if resolved_provider == "gemini":
        resolved_model = (model or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
    else:
        resolved_model = (model or os.getenv("LOCAL_GEMMA4_MODEL", "gemma-4")).strip()

    if not resolved_model:
        raise ValueError("Resolved model name cannot be empty.")
    return LLMConfig(provider=resolved_provider, model=resolved_model)


def resolve_llm_config_with_overrides(
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> LLMConfig:
    config = resolve_llm_config(provider, model)
    config.api_key = (api_key or "").strip() or None
    config.api_url = (api_url or "").strip() or None
    return config


def format_llm_label(config: LLMConfig) -> str:
    return f"{config.provider}:{config.model}"


def observation_to_prompt(obs: Observation) -> str:
    observation_json = obs.model_dump_json(exclude_none=True)
    return (
        "Return only JSON for an ETL remediation action. "
        "Schema: {\"decision\":\"fix|quarantine|promote|noop|finalize\","
        "\"threshold\":0.5,\"notes\":\"short reason\"}. "
        f"Observation: {observation_json}"
    )


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE).strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[len("```json") :].strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned[len("```") :].strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            cleaned = cleaned[start : end + 1].strip()
    return cleaned


def parse_action(text: str, provider: str) -> Action:
    cleaned = strip_code_fences(text)
    if not cleaned:
        raise ValueError(f"{provider} returned empty output.")
    payload = json.loads(cleaned)
    if not isinstance(payload, dict):
        raise ValueError(f"{provider} output must be a JSON object.")
    if "decision" not in payload:
        raise ValueError(f"{provider} output missing required decision field.")
    return Action(**payload)


def fallback_action(obs: Observation, error: str | None = None) -> Action:
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


def is_fatal_llm_error(error: str | None) -> bool:
    if not error:
        return False
    normalized = error.upper()
    return any(marker in normalized for marker in FATAL_ERROR_MARKERS)


def build_llm_client(config: LLMConfig) -> Any:
    if config.provider == "gemini":
        api_key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY missing. Set it in .env to enable Gemini inference.")
        return genai.Client(api_key=api_key)

    api_key = config.api_key or os.getenv("LOCAL_GEMMA4_API_KEY", "")
    return {
        "api_url": config.api_url or LOCAL_GEMMA4_API_URL,
        "api_key": api_key,
    }


def _generate_gemini_once(client: genai.Client, config: LLMConfig, obs: Observation) -> str:
    response = client.models.generate_content(
        model=config.model,
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


def _generate_local_once(client: dict[str, str], config: LLMConfig, obs: Observation) -> str:
    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return exactly one compact JSON object. "
                    "No markdown. No chain-of-thought. No explanation."
                ),
            },
            {
                "role": "user",
                "content": observation_to_prompt(obs),
            },
        ],
        "temperature": 0,
        "max_tokens": 512,
    }
    headers = {"Content-Type": "application/json"}
    if client.get("api_key"):
        headers["Authorization"] = f"Bearer {client['api_key']}"
    request = urllib.request.Request(
        client["api_url"],
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=LOCAL_TIMEOUT_SECONDS) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Local Gemma 4 HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Local Gemma 4 unavailable: {exc.reason}") from exc

    try:
        output_text = response_payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        raise RuntimeError(
            f"Local Gemma 4 response missing message content: {response_payload}"
        ) from exc
    if not output_text:
        raise RuntimeError("Local Gemma 4 returned empty output_text; cannot parse action.")
    return output_text


def _with_timeout(task_fn: Any, timeout_seconds: float) -> str:
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(task_fn)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError as exc:
        future.cancel()
        raise TimeoutError(f"LLM call timed out after {timeout_seconds:.0f}s.") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def get_llm_action(client: Any, config: LLMConfig, obs: Observation) -> Action:
    last_error: Exception | None = None
    max_retries = GEMINI_MAX_RETRIES if config.provider == "gemini" else LOCAL_MAX_RETRIES
    timeout_seconds = GEMINI_TIMEOUT_SECONDS if config.provider == "gemini" else LOCAL_TIMEOUT_SECONDS

    for attempt in range(max_retries + 1):
        try:
            if config.provider == "gemini":
                text = _with_timeout(lambda: _generate_gemini_once(client, config, obs), timeout_seconds)
            else:
                text = _with_timeout(lambda: _generate_local_once(client, config, obs), timeout_seconds)
            return parse_action(text, config.provider)
        except Exception as exc:
            last_error = exc
            if attempt >= max_retries:
                break

    provider_label = "Gemini" if config.provider == "gemini" else "Local Gemma 4"
    raise RuntimeError(
        truncate_text(last_error, 240) or f"{provider_label} action generation failed."
    )
