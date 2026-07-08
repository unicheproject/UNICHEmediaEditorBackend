"""audio.tts handler — routes to the configured inference provider.

Unlike image.caption/audio.transcribe, a successful synthesis produces an
audio file rather than plain JSON, so this overrides ProviderBackedHandler.run
to lift the provider's raw audio bytes out and persist them as a derived
asset via an OutputFile.
"""

from __future__ import annotations

from app.capabilities.context import HandlerResult, JobContext, OutputFile
from app.capabilities.handlers.base import ProviderBackedHandler
from app.models.enums import MediaType
from app.providers.base import InferenceRequest
from app.providers.factory import get_provider
from app.providers.openrouter import AUDIO_BYTES_KEY, AUDIO_FILENAME_KEY


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
        data = await provider.infer(request)
        audio_bytes = data.pop(AUDIO_BYTES_KEY, None)
        if audio_bytes is None:
            return HandlerResult(data=data, outputs=[])

        filename = data.pop(AUDIO_FILENAME_KEY, "speech.mp3")
        out_path = ctx.out_path(filename)
        out_path.write_bytes(audio_bytes)
        return HandlerResult(
            data=data,
            outputs=[OutputFile(path=out_path, filename=filename, media_type=MediaType.audio)],
        )
