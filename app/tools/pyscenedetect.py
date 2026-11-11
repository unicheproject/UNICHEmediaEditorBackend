"""PySceneDetect wrapper for deterministic shot/scene-boundary detection.

Unlike ffmpeg/imagemagick/realesrgan, PySceneDetect is a pip-installed Python
library, not a CLI producing a file we shell out to: its CLI only writes a CSV
report, so the Python API is used directly for structured (JSON-ready) results.
Detection is synchronous/CPU-bound, so it's offloaded to a thread via
asyncio.to_thread to avoid blocking the worker's event loop.
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.core.errors import ToolError
from app.tools import runner
from app.tools.demuxers import UNTRUSTED_INPUT

# ContentDetector default threshold (0-255 HSV+edge content-change score).
DEFAULT_THRESHOLD = 27.0


def _detect_sync(path: str, threshold: float) -> list[dict[str, Any]]:
    from scenedetect import ContentDetector, detect

    try:
        scenes = detect(path, ContentDetector(threshold=threshold), start_in_scene=True)
    except Exception as exc:
        raise ToolError(f"video.shot.detect failed: {exc}") from exc

    return [
        {
            "index": i,
            "start": round(start.seconds, 3),
            "end": round(end.seconds, 3),
            "start_frame": start.frame_num,
            "end_frame": end.frame_num,
        }
        for i, (start, end) in enumerate(scenes)
    ]


async def _preflight(src: str) -> None:
    """Open the file with the restricted demuxers before OpenCV opens it.

    OpenCV's own FFmpeg ignores format_whitelist (measured: it decoded an HLS
    playlist disguised as .mp4, with or without OPENCV_FFMPEG_CAPTURE_OPTIONS).
    FFmpeg picks the demuxer for the contents and then applies the allowlist,
    so a file this probe opens is one FFmpeg reads as an allowed container.
    Asset files are not rewritten in place, so OpenCV opens the same bytes.
    """
    try:
        out = await runner.run_stdout(
            ["ffprobe", "-v", "error", "-select_streams", "V:0",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0",
             *UNTRUSTED_INPUT, "-i", src],
            timeout=60,
        )
    except ToolError:
        raise ToolError("The video could not be opened") from None
    if out.strip() != "video":
        raise ToolError("The file has no video to detect shots in")


async def detect_shots(src: str, threshold: float | None = None) -> list[dict[str, Any]]:
    threshold = DEFAULT_THRESHOLD if threshold is None else threshold
    await _preflight(src)
    return await asyncio.to_thread(_detect_sync, src, threshold)
