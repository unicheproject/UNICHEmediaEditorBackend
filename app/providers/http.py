"""HTTP inference provider — calls a configured hosted inference endpoint.

Capability -> endpoint path mapping comes from settings, so deployments can
point each capability at a real hosted endpoint via environment variables
without any code change in routes or job-creation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

from app.core.config import Settings
from app.core.errors import ProviderError
from app.core.logging import get_logger
from app.providers.base import BaseInferenceProvider, InferenceRequest

logger = get_logger(__name__)


class HTTPInferenceProvider(BaseInferenceProvider):
    name = "http"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._paths: dict[str, str] = {
            "image.caption": settings.inference_image_caption_path,
            "audio.transcribe": settings.inference_audio_transcribe_path,
        }

    def _endpoint_for(self, capability_id: str) -> str:
        base = self._settings.inference_base_url.rstrip("/")
        if not base:
            raise ProviderError("INFERENCE_BASE_URL is not configured")
        path = self._paths.get(capability_id)
        if not path:
            raise ProviderError(
                f"No HTTP endpoint path configured for capability '{capability_id}'"
            )
        return f"{base}/{path.lstrip('/')}"

    async def infer(self, request: InferenceRequest) -> dict[str, Any]:
        url = self._endpoint_for(request.capability_id)
        headers: dict[str, str] = {}
        if self._settings.inference_api_key:
            headers["Authorization"] = f"Bearer {self._settings.inference_api_key}"

        data = {"input": json.dumps(request.payload)}
        files: dict[str, Any] = {}
        open_file = None
        try:
            if request.file_path and Path(request.file_path).exists():
                open_file = open(request.file_path, "rb")  # noqa: SIM115
                files["file"] = (
                    request.asset_meta.get("original_filename", "upload"),
                    open_file,
                    request.asset_meta.get("mime_type", "application/octet-stream"),
                )

            async with httpx.AsyncClient(
                timeout=self._settings.inference_timeout_seconds
            ) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    data=data,
                    files=files or None,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Inference request timed out: {url}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Inference request failed: {exc}") from exc
        finally:
            if open_file is not None:
                open_file.close()

        if response.status_code >= 400:
            raise ProviderError(
                f"Inference endpoint returned {response.status_code}: "
                f"{response.text[:500]}"
            )

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError("Inference endpoint returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise ProviderError("Inference endpoint returned non-object JSON")

        payload.setdefault("provider", self.name)
        return payload
