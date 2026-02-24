"""
CellAtria 2.0 — Shared Status Logger
=======================================
Thread-safe, module-level singleton that pipeline agents push
health / status events to.  The Streamlit UI reads from this to
render the "Background Checks" panel.

Usage (from any agent or utility)
---------------------------------
    from core.status_logger import status_log

    status_log.push("groq", "ok",   "Llama-3.3-70B responded in 1.2 s")
    status_log.push("groq", "warn", "Rate limited — retrying in 8 s …")
    status_log.push("e2b",  "error","E2B_API_KEY not set")

    summary = status_log.get_summary()
    # → {"groq": {"level": "ok", "message": "..."}, "e2b": {...}}
"""

from __future__ import annotations

import threading
from collections import deque
from datetime import datetime
from typing import Any


# Valid service names
SERVICES = ("groq", "e2b", "ftp", "pipeline")

# Valid severity levels (ascending severity)
LEVELS = ("ok", "warn", "error")


class _StatusEvent:
    """Single event pushed by an agent or utility."""
    __slots__ = ("service", "level", "message", "timestamp")

    def __init__(self, service: str, level: str, message: str) -> None:
        self.service = service
        self.level = level
        self.message = message
        self.timestamp = datetime.now().strftime("%H:%M:%S")

    def to_dict(self) -> dict[str, str]:
        return {
            "service": self.service,
            "level": self.level,
            "message": self.message,
            "timestamp": self.timestamp,
        }


class StatusLogger:
    """Thread-safe status logger with bounded event history."""

    def __init__(self, max_events: int = 50) -> None:
        self._lock = threading.Lock()
        self._events: deque[_StatusEvent] = deque(maxlen=max_events)
        # Latest status per service
        self._latest: dict[str, _StatusEvent] = {}

    # ── Write (called from background threads / agents) ──────────
    def push(self, service: str, level: str, message: str) -> None:
        """Record a status event.

        Parameters
        ----------
        service : str
            One of ``"groq"``, ``"e2b"``, ``"ftp"``, ``"pipeline"``.
        level : str
            ``"ok"`` | ``"warn"`` | ``"error"``
        message : str
            Human-readable description.
        """
        event = _StatusEvent(service, level, message)
        with self._lock:
            self._events.append(event)
            self._latest[service] = event

    def clear(self) -> None:
        """Reset all events (e.g. when starting a new pipeline run)."""
        with self._lock:
            self._events.clear()
            self._latest.clear()

    # ── Read (called from Streamlit main thread) ─────────────────
    def get_summary(self) -> dict[str, dict[str, str]]:
        """Return the latest status per service.

        Returns
        -------
        dict
            ``{"groq": {"level": "ok", "message": "...", "timestamp": "..."}, ...}``
        """
        with self._lock:
            return {
                svc: evt.to_dict()
                for svc, evt in self._latest.items()
            }

    def get_recent(self, n: int = 10) -> list[dict[str, str]]:
        """Return the last *n* events as a list of dicts."""
        with self._lock:
            items = list(self._events)
        return [e.to_dict() for e in items[-n:]]

    def get_all_events(self) -> list[dict[str, str]]:
        """Return all recorded events as a list of dicts."""
        with self._lock:
            return [e.to_dict() for e in self._events]


# ── Module-level singleton ───────────────────────────────────────
status_log = StatusLogger()
