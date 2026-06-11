"""Async subprocess runner for local CLI tools.

Isolates all subprocess invocation so handlers stay declarative. Any non-zero
exit, missing binary, or timeout surfaces as ToolError, which execute_job
records as a job failure.
"""

from __future__ import annotations

import asyncio

from app.core.errors import ToolError
from app.core.logging import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 600.0


async def _exec(args: list[str], timeout: float) -> tuple[str, str]:
    """Run a command to completion. Returns (stdout, stderr).

    Raises ToolError on missing binary, non-zero exit, or timeout.
    """
    logger.info("Running tool: %s", " ".join(args))
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ToolError(f"Tool not found: {args[0]}") from exc

    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except TimeoutError as exc:
        proc.kill()
        await proc.wait()
        raise ToolError(f"Tool '{args[0]}' timed out after {timeout}s") from exc

    stdout = stdout_b.decode("utf-8", errors="replace")
    stderr = stderr_b.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        tail = stderr.strip().splitlines()[-15:]
        raise ToolError(
            f"Tool '{args[0]}' failed (exit {proc.returncode}): " + " | ".join(tail)
        )
    return stdout, stderr


async def run(args: list[str], *, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Run a command, returning captured stderr (tools log progress there)."""
    _stdout, stderr = await _exec(args, timeout)
    return stderr


async def run_stdout(args: list[str], *, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Run a command, returning captured stdout (for query tools like ffprobe)."""
    stdout, _stderr = await _exec(args, timeout)
    return stdout
