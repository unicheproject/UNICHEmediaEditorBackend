"""Capability handler interface.

A handler turns a JobContext into a HandlerResult. Two base families:
- ProviderBackedHandler: routes to the configured inference provider (JSON out).
- LocalToolHandler: runs a deterministic local CLI tool (file output).

Provider selection / subprocess details live below the handler boundary, so the
API and job-creation code never change when capabilities are added or swapped.
"""

from __future__ import annotations

import abc

from app.capabilities.context import HandlerResult, JobContext
from app.providers.base import InferenceRequest
from app.providers.factory import get_provider


class CapabilityHandler(abc.ABC):
    capability_id: str

    @abc.abstractmethod
    async def run(self, ctx: JobContext) -> HandlerResult:
        raise NotImplementedError


class ProviderBackedHandler(CapabilityHandler):
    """Handler that routes to the configured inference provider."""

    async def run(self, ctx: JobContext) -> HandlerResult:
        provider = get_provider()
        request = InferenceRequest(
            capability_id=ctx.capability_id,
            payload=ctx.params,
            file_path=ctx.input_path,
            asset_meta=ctx.input_asset_meta,
        )
        data = await provider.infer(request)
        return HandlerResult(data=data, outputs=[])


class LocalToolHandler(CapabilityHandler):
    """Base for deterministic local-tool handlers (ffmpeg / imagemagick).

    Subclasses implement `process` and may write files into `ctx.work_dir`,
    returning them as OutputFile entries on the HandlerResult.
    """

    @abc.abstractmethod
    async def process(self, ctx: JobContext) -> HandlerResult:
        raise NotImplementedError

    async def run(self, ctx: JobContext) -> HandlerResult:
        return await self.process(ctx)
