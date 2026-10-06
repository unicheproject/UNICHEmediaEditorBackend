"""audio.tts handler — routes to the configured inference provider.

Unlike image.caption/audio.transcribe, a successful synthesis produces an
audio file rather than plain JSON, so this overrides ProviderBackedHandler.run
to lift the provider's raw audio bytes out and persist them as a derived
asset via an OutputFile.
"""

from __future__ import annotations

from pathlib import Path

from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.capabilities.handlers.base import ProviderBackedHandler
from app.core.errors import ProviderError
from app.models.enums import MediaType
from app.providers.base import InferenceRequest
from app.providers.factory import get_provider
from app.providers.openrouter import AUDIO_BYTES_KEY, AUDIO_FILENAME_KEY
from app.providers.validation import require_success_object
from app.services.media_metadata import MetadataProbeError, probe_media


class AudioTtsHandler(ProviderBackedHandler):
    capability_id = "audio.tts"

    async def run(self, ctx: JobContext) -> HandlerResult:
        provider = get_provider()
        request = InferenceRequest(
            capability_id=ctx.capability_id,
            payload=ctx.params,
            file_path=ctx.input_path,
            asset_meta=ctx.input_asset_meta,
        )
        data = require_success_object(await provider.infer(request))
        audio_bytes = data.pop(AUDIO_BYTES_KEY, None)
        if not isinstance(audio_bytes, bytes) or not audio_bytes:
            raise ProviderError("Speech synthesis returned no audio output")

        filename = data.pop(AUDIO_FILENAME_KEY, "speech.mp3")
        if not isinstance(filename, str) or Path(filename).name != filename or not filename:
            raise ProviderError("Speech synthesis returned an invalid audio filename")
        out_path = ctx.out_path(filename)
        try:
            out_path.write_bytes(audio_bytes)
            metadata = await probe_media(out_path, MediaType.audio)
        except (OSError, MetadataProbeError) as exc:
            raise ProviderError("Speech synthesis returned invalid or unprobeable audio") from exc
        if (
            not metadata.has_audio_stream
            or metadata.duration_seconds is None
            or metadata.duration_seconds <= 0
        ):
            raise ProviderError("Speech synthesis returned no non-empty audio stream")
        return HandlerResult(
            data=data,
            outputs=[OutputFile(path=out_path, filename=filename, media_type=MediaType.audio)],
        )
