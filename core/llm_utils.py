"""
CellAtria 2.0 — Rate-Limit-Aware LLM Wrapper (v2)
====================================================
Centralised ``call_llm()`` function that every agent uses instead
of creating raw ``ChatGroq`` instances.  Provides:

* **Capped exponential backoff** on 429 (rate-limit) errors.
  MAX_RETRIES=2, BASE_DELAY=2s, MAX_DELAY=8s — worst-case 10s.
* **In-memory response cache** for identical prompts to avoid
  redundant API calls on reruns.
* **Live status reporting** via ``core.status_logger``.
* **Graceful degradation** — returns ``None`` after max retries
  instead of crashing the pipeline.

Usage
-----
    from core.llm_utils import call_llm

    response = call_llm(
        system_prompt="You are a helpful assistant.",
        user_prompt="Extract accessions from the following text...",
        max_tokens=512,
        temperature=0.0,
    )
    if response is None:
        # All retries exhausted — handle gracefully
        ...
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from core.status_logger import status_log

# ── Logging ──────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Configuration (tuned for speed) ─────────────────────────────
MAX_RETRIES = 2               # was 4 — now fail fast
BASE_BACKOFF_SECONDS = 2      # was 4 — backoff: 2s → 4s
MAX_BACKOFF_SECONDS = 8       # hard cap on any single wait
DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_CACHE_SIZE = 100          # cap cache entries to prevent memory bloat

# ── In-Memory LLM Cache ─────────────────────────────────────────
# Maps hash(system_prompt + user_prompt) → response string.
# Prevents re-calling the API for identical prompts during the
# same process lifetime (survives Streamlit reruns).
_LLM_CACHE: dict[str, str] = {}


def _cache_key(system_prompt: str, user_prompt: str, model: str) -> str:
    """Generate a deterministic cache key from the prompt pair."""
    raw = f"{model}||{system_prompt}||{user_prompt}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def clear_llm_cache() -> None:
    """Flush the entire in-memory LLM response cache."""
    _LLM_CACHE.clear()
    logger.info("LLM response cache cleared.")


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect Groq 429 / rate-limit errors from the exception."""
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate_limit" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "resource_exhausted" in msg
    )


def call_llm(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 4096,
    caller: str = "agent",
    use_cache: bool = True,
) -> str | None:
    """Call Groq LLM with capped backoff and caching.

    Parameters
    ----------
    system_prompt : str
        System-level instruction.
    user_prompt : str
        The user/human message.
    model : str
        Groq model identifier.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens in the response.
    caller : str
        Name of the calling agent (for logging clarity).
    use_cache : bool
        If True, return cached response for identical prompts.

    Returns
    -------
    str | None
        The LLM response text, or ``None`` if all retries failed.
    """
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        status_log.push("groq", "error", "GROQ_API_KEY is not set")
        logger.error("[%s] GROQ_API_KEY is not set — cannot call LLM.", caller)
        return None

    # ── Check cache first ────────────────────────────────────────
    if use_cache:
        key = _cache_key(system_prompt, user_prompt, model)
        cached = _LLM_CACHE.get(key)
        if cached is not None:
            status_log.push("groq", "ok", f"{caller}: cache hit (skipping API call)")
            logger.info("[%s] LLM cache hit — returning cached response.", caller)
            return cached

    # ── Retry loop ───────────────────────────────────────────────
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            llm = ChatGroq(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=api_key,
            )

            t0 = time.time()
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            elapsed = time.time() - t0

            result_text = response.content.strip()

            status_log.push(
                "groq", "ok",
                f"{caller}: {model} responded in {elapsed:.1f}s",
            )
            logger.info(
                "[%s] LLM call succeeded (attempt %d, %.1fs).",
                caller, attempt, elapsed,
            )

            # ── Store in cache (with size cap) ──────────────────
            if use_cache:
                if len(_LLM_CACHE) >= MAX_CACHE_SIZE:
                    # Evict oldest entry (insertion-order, Python 3.7+)
                    _oldest = next(iter(_LLM_CACHE))
                    del _LLM_CACHE[_oldest]
                _LLM_CACHE[key] = result_text

            return result_text

        except Exception as exc:
            if _is_rate_limit_error(exc):
                wait = min(
                    BASE_BACKOFF_SECONDS * (2 ** (attempt - 1)),
                    MAX_BACKOFF_SECONDS,
                )
                msg = (
                    f"{caller}: Rate limited (attempt {attempt}/{MAX_RETRIES}) "
                    f"— retrying in {wait}s"
                )
                status_log.push("groq", "warn", msg)
                logger.warning("[%s] %s", caller, msg)
                time.sleep(wait)
            else:
                # Non-rate-limit error — don't retry
                error_msg = f"{caller}: LLM error — {type(exc).__name__}: {exc}"
                status_log.push("groq", "error", error_msg)
                logger.error("[%s] %s", caller, error_msg)
                return None

    # All retries exhausted
    exhausted_msg = (
        f"{caller}: Groq rate limit — all {MAX_RETRIES} retries exhausted"
    )
    status_log.push("groq", "error", exhausted_msg)
    logger.error("[%s] %s", caller, exhausted_msg)
    return None
