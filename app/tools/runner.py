"""Async subprocess runner for local CLI tools.

Isolates all subprocess invocation so handlers stay declarative. Any non-zero
exit, missing binary, or timeout surfaces as ToolError, which execute_job
records as a job failure.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from app.core.errors import ToolError
from app.core.logging import get_logger
from app.tools.process_policy import IO_CHUNK_BYTES, media_process

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 600.0
# Five current stdout consumers request only small ffprobe JSON/CSV subsets;
# match the existing metadata-probe cap rather than allowing arbitrary output.
MAX_STDOUT_BYTES = 1024 * 1024
MAX_STDERR_BYTES = 16 * 1024


async def _stdout(stream: asyncio.StreamReader) -> bytes:
    output = bytearray()
    while chunk := await stream.read(IO_CHUNK_BYTES):
        if len(output) + len(chunk) > MAX_STDOUT_BYTES:
            raise ToolError("Media processing output exceeded its limit.")
        output.extend(chunk)
    return bytes(output)


async def _stderr(stream: asyncio.StreamReader, tail: bytearray) -> None:
    while chunk := await stream.read(IO_CHUNK_BYTES):
        tail.extend(chunk)
        del tail[:-MAX_STDERR_BYTES]


async def _exec(args: list[str], timeout: float, *, capture_stdout: bool) -> tuple[str, str]:
    """Drain bounded I/O; diagnostics stay server-side, never in ToolError."""
    executable = Path(args[0]).name
    tool = (executable if executable in {"ffmpeg", "ffprobe", "convert", "identify"}
            else "media tool")
    tail = bytearray()
    try:
        async with media_process(
            args, stdout=asyncio.subprocess.PIPE if capture_stdout else asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
        ) as proc:
            assert proc.stderr is not None
            tasks: list[asyncio.Task] = [asyncio.create_task(_stderr(proc.stderr, tail))]
            output_task = None
            if capture_stdout:
                assert proc.stdout is not None
                output_task = asyncio.create_task(_stdout(proc.stdout))
                tasks.append(output_task)
            tasks.append(asyncio.create_task(proc.wait()))
            try:
                async with asyncio.timeout(timeout):
                    await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            if proc.returncode != 0:
                logger.warning("%s failed (exit %s)", tool, proc.returncode)
                logger.debug("%s diagnostic tail: %r", tool, bytes(tail))
                raise ToolError("Media processing failed.")
            output = output_task.result() if output_task else b""
    except TimeoutError as exc:
        logger.warning("%s exceeded execution deadline", tool)
        raise ToolError("Media processing timed out.") from exc
    except OSError as exc:
        logger.warning("%s could not be launched (%s)", tool, type(exc).__name__)
        raise ToolError("Media processing tool unavailable.") from exc
    except ToolError as exc:
        if exc.message == "Media processing output exceeded its limit.":
            logger.warning("%s exceeded stdout capture limit", tool)
        raise
    except asyncio.CancelledError:
        logger.info("%s execution cancelled", tool)
        raise
    return output.decode("utf-8", errors="replace"), tail.decode("utf-8", errors="replace")


async def run(args: list[str], *, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Discard stdout; return only a bounded stderr tail for internal callers."""
    _stdout, stderr = await _exec(args, timeout, capture_stdout=False)
    return stderr


async def run_stdout(args: list[str], *, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Return complete stdout up to 1 MiB; overflow fails, never truncates."""
    stdout, _stderr = await _exec(args, timeout, capture_stdout=True)
    return stdout
