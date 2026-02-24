"""
CellAtria 2.0 - E2B Sandbox Manager
===================================
Provides `run_in_sandbox()` for executing generated Python scripts in an
E2B sandbox. Supports both SDK styles:
1. `sandbox.run_code(...)`
2. `sandbox.commands.run(...)`
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
from typing import Any

from core.status_logger import status_log

logger = logging.getLogger(__name__)

E2B_API_KEY = os.environ.get("E2B_API_KEY", "")
E2B_TEMPLATE_ID = os.environ.get("E2B_TEMPLATE_ID", "cellatria-sandbox")
E2B_TIMEOUT_SECONDS = int(os.environ.get("E2B_TIMEOUT_SECONDS", "300"))
SANDBOX_CREATION_TIMEOUT = int(
    os.environ.get("E2B_SANDBOX_CREATION_TIMEOUT", "90")
)
REMOTE_SCRIPT_PATH = os.environ.get(
    "E2B_REMOTE_SCRIPT_PATH", "/tmp/cellatria_exec.py"
)


def check_e2b_health() -> dict[str, str]:
    """Proactive health check for E2B sandbox availability."""
    if not E2B_API_KEY:
        msg = "E2B_API_KEY not set - sandbox execution disabled"
        status_log.push("e2b", "error", msg)
        return {"level": "error", "message": msg}

    try:
        from e2b import Sandbox  # type: ignore[import-untyped]
    except ImportError:
        try:
            import e2b_code_interpreter  # noqa: F401
            msg = (
                "e2b package missing, but e2b_code_interpreter is installed. "
                "Install e2b for Sandbox API compatibility."
            )
            status_log.push("e2b", "warn", msg)
            return {"level": "warn", "message": msg}
        except ImportError:
            msg = "E2B SDK not installed - pip install e2b"
            status_log.push("e2b", "warn", msg)
            return {"level": "warn", "message": msg}

    capabilities: list[str] = []
    if hasattr(Sandbox, "run_code"):
        capabilities.append("run_code")
    if hasattr(Sandbox, "commands"):
        capabilities.append("commands.run")

    if not capabilities:
        msg = "E2B SDK detected, but no supported execution API found."
        status_log.push("e2b", "warn", msg)
        return {"level": "warn", "message": msg}

    cap_str = ", ".join(capabilities)
    msg = f"E2B SDK ready - Template: {E2B_TEMPLATE_ID} - APIs: {cap_str}"
    status_log.push("e2b", "ok", msg)
    return {"level": "ok", "message": msg}


def _create_sandbox() -> tuple[Any | None, str]:
    """Create a sandbox instance with compatibility fallbacks."""
    try:
        from e2b import Sandbox  # type: ignore[import-untyped]
    except ImportError:
        return None, "e2b is not installed."

    constructors = [
        lambda: Sandbox.create(
            template=E2B_TEMPLATE_ID,
            api_key=E2B_API_KEY,
            timeout=E2B_TIMEOUT_SECONDS,
        ),
        lambda: Sandbox.create(
            template=E2B_TEMPLATE_ID,
            timeout=E2B_TIMEOUT_SECONDS,
        ),
        lambda: Sandbox.create(
            api_key=E2B_API_KEY,
            timeout=E2B_TIMEOUT_SECONDS,
        ),
        lambda: Sandbox.create(timeout=E2B_TIMEOUT_SECONDS),
    ]

    last_error = ""
    for idx, ctor in enumerate(constructors, start=1):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(ctor)
            sandbox = future.result(timeout=SANDBOX_CREATION_TIMEOUT)
            logger.info("E2B sandbox created (constructor variant %d).", idx)
            return sandbox, ""
        except concurrent.futures.TimeoutError:
            last_error = (
                f"constructor variant {idx} timed out after "
                f"{SANDBOX_CREATION_TIMEOUT}s"
            )
            logger.warning("E2B constructor variant %d timed out.", idx)
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "E2B constructor variant %d failed: %s",
                idx,
                last_error,
            )
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    return None, f"All E2B constructors failed. Last error: {last_error}"


def _log_item_to_text(item: Any) -> str:
    """Normalize E2B log item types into plain text."""
    if isinstance(item, str):
        return item
    text_attr = getattr(item, "text", None)
    if isinstance(text_attr, str):
        return text_attr
    return str(item)


def _run_with_run_code(sandbox: Any, script: str) -> dict[str, Any]:
    """Execute using SDKs that expose sandbox.run_code()."""
    execution = sandbox.run_code(script)

    stdout_parts: list[str] = []
    stderr_parts: list[str] = []

    logs = getattr(execution, "logs", None)
    if logs is not None:
        for item in getattr(logs, "stdout", []) or []:
            stdout_parts.append(_log_item_to_text(item))
        for item in getattr(logs, "stderr", []) or []:
            stderr_parts.append(_log_item_to_text(item))

    stdout = "\n".join(stdout_parts)
    stderr = "\n".join(stderr_parts)

    exec_error = getattr(execution, "error", None)
    if exec_error:
        err_name = getattr(exec_error, "name", "ExecutionError")
        err_value = getattr(exec_error, "value", str(exec_error))
        err_trace = getattr(exec_error, "traceback", "")
        error_msg = f"{err_name}: {err_value}\n{err_trace}".strip()
        return {
            "stdout": stdout,
            "stderr": stderr,
            "success": False,
            "error": error_msg,
        }

    return {
        "stdout": stdout,
        "stderr": stderr,
        "success": True,
        "error": "",
    }


def _run_with_commands(sandbox: Any, script: str) -> dict[str, Any]:
    """Execute using SDKs that expose sandbox.commands.run()."""
    sandbox.files.write(REMOTE_SCRIPT_PATH, script)
    try:
        result = sandbox.commands.run(
            f"python {REMOTE_SCRIPT_PATH}",
            timeout=E2B_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "stdout": "",
            "stderr": "",
            "success": False,
            "error": str(exc),
        }

    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    exit_code = int(getattr(result, "exit_code", 1))
    command_error = getattr(result, "error", None)

    if command_error or exit_code != 0:
        detail = str(command_error).strip() if command_error else ""
        if stderr.strip():
            detail = f"{detail}\n{stderr}".strip()
        if not detail:
            detail = f"Command failed with exit code {exit_code}"
        return {
            "stdout": stdout,
            "stderr": stderr,
            "success": False,
            "error": detail,
        }

    return {
        "stdout": stdout,
        "stderr": stderr,
        "success": True,
        "error": "",
    }


def run_in_sandbox(script: str) -> dict[str, Any]:
    """Execute Python source code inside an E2B sandbox."""
    if not E2B_API_KEY:
        msg = "E2B_API_KEY environment variable is not set."
        logger.error(msg)
        return {"stdout": "", "stderr": "", "success": False, "error": msg}

    sandbox, init_error = _create_sandbox()
    if sandbox is None:
        logger.error("E2B sandbox creation failed: %s", init_error)
        status_log.push("e2b", "error", f"Sandbox init failed: {init_error[:120]}")
        return {
            "stdout": "",
            "stderr": "",
            "success": False,
            "error": init_error,
        }

    try:
        logger.info("Executing script in sandbox (%d chars).", len(script))
        status_log.push("e2b", "ok", "Executing script in sandbox...")

        if hasattr(sandbox, "run_code"):
            result = _run_with_run_code(sandbox, script)
        elif hasattr(sandbox, "commands"):
            result = _run_with_commands(sandbox, script)
        else:
            result = {
                "stdout": "",
                "stderr": "",
                "success": False,
                "error": "Sandbox object has no supported execution method.",
            }

        if result["success"]:
            status_log.push("e2b", "ok", "Sandbox execution succeeded")
        else:
            status_log.push("e2b", "warn", "Sandbox execution failed")
        return result

    except Exception as exc:  # noqa: BLE001
        error_msg = f"E2B sandbox exception: {type(exc).__name__}: {exc}"
        logger.exception(error_msg)
        status_log.push("e2b", "error", error_msg[:120])
        return {
            "stdout": "",
            "stderr": "",
            "success": False,
            "error": error_msg,
        }
    finally:
        try:
            if hasattr(sandbox, "kill"):
                sandbox.kill()
                logger.info("Sandbox closed via kill().")
        except Exception:  # noqa: BLE001
            logger.debug("Sandbox cleanup failed - ignoring.")
