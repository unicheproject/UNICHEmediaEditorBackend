"""Linux media-process launch/lifecycle policy; not a filesystem/network sandbox."""

from __future__ import annotations

import asyncio
import os
import signal
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from app.core.logging import get_logger

logger = get_logger(__name__)
TOOL_PATH = "/usr/local/bin:/usr/bin:/bin"
TERMINATION_GRACE_SECONDS = 0.25
REAP_TIMEOUT_SECONDS = 1.0
IO_CHUNK_BYTES = 64 * 1024
# Retain ownership if process creation acknowledges late after cancellation.
_pending_cleanup: set[asyncio.Task] = set()


def media_environment(workspace: str) -> dict[str, str]:
    # Committed runtime uses system libraries/fonts. No inherited loader/config
    # overrides, application secrets or executable search paths.
    return {
        "PATH": TOOL_PATH, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "HOME": workspace, "TMPDIR": workspace,
    }


async def _discard(stream: asyncio.StreamReader) -> None:
    while await stream.read(IO_CHUNK_BYTES):
        pass


async def terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """After caller stops reading, signal the group and drain without retaining.

    A reaped leader does not imply descendants exited. Reaping is bounded even
    for a child that escaped the process group (not a complete sandbox).
    """
    drains = [asyncio.create_task(_discard(stream))
              for stream in (proc.stdout, proc.stderr) if stream is not None]
    try:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            pass
        else:
            await asyncio.sleep(TERMINATION_GRACE_SECONDS)
            with suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
        try:
            await asyncio.wait_for(proc.wait(), REAP_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning("Media process reap exceeded cleanup deadline")
    finally:
        for task in drains:
            task.cancel()
        await asyncio.gather(*drains, return_exceptions=True)


async def _cleanup(proc: asyncio.subprocess.Process) -> None:
    # Repeated cancellation cannot interrupt the bounded cleanup operation.
    task = asyncio.create_task(terminate_process_group(proc))
    cancelled = False
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
    await task
    if cancelled:
        raise asyncio.CancelledError


def _late_cleanup_finished(task: asyncio.Task) -> None:
    _pending_cleanup.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Late media process cleanup failed")


def _late_launch_finished(task: asyncio.Task[asyncio.subprocess.Process]) -> None:
    _pending_cleanup.discard(task)
    if task.cancelled() or task.exception() is not None:
        return
    cleanup = asyncio.create_task(_cleanup(task.result()))
    _pending_cleanup.add(cleanup)
    cleanup.add_done_callback(_late_cleanup_finished)


@asynccontextmanager
async def media_process(
    args: list[str], *, stdout: int, stderr: int, limit: int = IO_CHUNK_BYTES,
) -> AsyncIterator[asyncio.subprocess.Process]:
    """Argv only, private scratch cwd, explicit env/stdin/session for every tool."""
    with tempfile.TemporaryDirectory(prefix="media-process-", dir="/tmp") as workspace:
        spawn = asyncio.create_task(asyncio.create_subprocess_exec(
            *args, stdin=asyncio.subprocess.DEVNULL, stdout=stdout, stderr=stderr,
            limit=limit, env=media_environment(workspace), cwd=workspace,
            start_new_session=True,
        ))
        try:
            proc = await asyncio.shield(spawn)
        except asyncio.CancelledError:
            # Do not lose a newly spawned PID or wait indefinitely for creation
            # acknowledgement. A late acknowledgement keeps supervised cleanup.
            deadline = asyncio.get_running_loop().time() + REAP_TIMEOUT_SECONDS
            while not spawn.done():
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    await asyncio.wait_for(asyncio.shield(spawn), remaining)
                except asyncio.CancelledError:
                    continue
                except Exception:  # noqa: BLE001 - preserve authoritative cancellation
                    break
            if not spawn.done():
                _pending_cleanup.add(spawn)
                spawn.add_done_callback(_late_launch_finished)
                logger.warning("Media process creation acknowledgement was delayed")
                raise
            if not spawn.cancelled() and spawn.exception() is None:
                await _cleanup(spawn.result())
            raise
        try:
            yield proc
        finally:
            await _cleanup(proc)
