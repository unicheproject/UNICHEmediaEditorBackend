"""subtitle.autogenerate — timed SRT subtitles from speech via the inference provider.

The input's first soundtrack is extracted to a small mono M4A, transcribed with
segment timestamps (the same provider route as audio.transcribe), validated,
and published as a derived subtitle Asset.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any

from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.capabilities.handlers._util import input_stem, require_input
from app.capabilities.handlers.base import ProviderBackedHandler
from app.core.errors import ProviderError, ValidationError
from app.models.enums import MediaType
from app.providers.validation import require_transcript
from app.tools import ffmpeg


@dataclasses.dataclass(frozen=True)
class Cue:
    start: float
    end: float
    text: str


def parse_segments(raw: object) -> list[Cue]:
    """Validate provider segments into ordered, non-empty cues."""
    if not isinstance(raw, list):
        return []
    cues = []
    for item in raw:
        if not isinstance(item, dict):
            raise ProviderError("Transcription returned malformed timed segments")
        start, end, text = item.get("start"), item.get("end"), item.get("text")
        if (
            isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, (int, float)) or not isinstance(end, (int, float))
            or not math.isfinite(start) or not math.isfinite(end)
            or start < 0 or end <= start or not isinstance(text, str)
        ):
            raise ProviderError("Transcription returned malformed timed segments")
        if text.strip():
            cues.append(Cue(float(start), float(end), " ".join(text.split())))
    return sorted(cues, key=lambda cue: (cue.start, cue.end))


def _timestamp(seconds: float) -> str:
    millis = round(seconds * 1000)
    hours, rest = divmod(millis, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, ms = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks = [
        f"{index}\n{_timestamp(cue.start)} --> {_timestamp(cue.end)}\n{cue.text}\n"
        for index, cue in enumerate(cues, start=1)
    ]
    return "\n".join(blocks)


class SubtitleAutogenerateHandler(ProviderBackedHandler):
    capability_id = "subtitle.autogenerate"

    async def run(self, ctx: JobContext) -> HandlerResult:
        src = require_input(ctx)
        stem = input_stem(ctx)
        speech = ctx.out_path(f"{stem}_speech.m4a")
        await ffmpeg.extract_speech_audio(src, str(speech))
        provider_ctx = dataclasses.replace(
            ctx,
            input_path=str(speech),
            input_asset_meta={
                **ctx.input_asset_meta,
                "original_filename": speech.name,
                "mime_type": "audio/mp4",
                "media_type": MediaType.audio.value,
            },
        )
        data: dict[str, Any] = dict((await super().run(provider_ctx)).data)
        require_transcript(data)
        cues = parse_segments(data.get("segments"))
        if not cues:
            if data["text"].strip():
                raise ProviderError(
                    "Transcription returned no timed segments; the configured "
                    "provider may not support timestamps"
                )
            raise ValidationError("No speech was detected to create subtitles from")

        filename = f"{stem}.srt"
        out = ctx.out_path(filename)
        Path(out).write_text(to_srt(cues), encoding="utf-8")
        result = {
            "operation": "subtitle.autogenerate",
            "language": data.get("language"),
            "segment_count": len(cues),
            "provider": data.get("provider"),
        }
        if data.get("model"):
            result["model"] = data["model"]
        return HandlerResult(
            data=result,
            outputs=[OutputFile(path=out, filename=filename, media_type=MediaType.subtitle)],
        )
