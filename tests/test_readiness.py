"""Readiness is canonical, independent of applicability, and enforced before work."""

from __future__ import annotations

import sys
import uuid
from types import SimpleNamespace

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select

from app.agent.catalog import build_catalog
from app.agent.executor import execute_plan
from app.agent.planner import PlanValidationError, validate_plan
from app.agent.schemas import Plan
from app.api.v1.jobs import get_enqueuer
from app.capabilities import registry
from app.capabilities.context import JobContext
from app.capabilities.handlers import base as handler_base
from app.capabilities.handlers.audio_tts import AudioTtsHandler
from app.capabilities.handlers.not_implemented import NotImplementedHandler
from app.capabilities.readiness import (
    capability_read,
    require_executable,
    resolve_availability,
    resolve_readiness,
)
from app.core.config import Settings, settings
from app.core.database import async_session_factory
from app.core.errors import CapabilityUnavailableError, ProviderError, ToolError
from app.main import app
from app.models.agent import AgentPlan, AgentPlanAttempt, AgentSession
from app.models.asset import Asset
from app.models.enums import AgentPlanStatus, JobStatus
from app.models.job import Job
from app.providers.base import BaseInferenceProvider, InferenceRequest
from app.schemas.capability import (
    DeploymentReadiness,
    DeploymentRequirement,
    ImplementationReadiness,
    ReadinessReason,
)
from app.schemas.job import JobCreate
from app.services import agent as agent_svc
from app.services.jobs import create_job, execute_job
from app.tools import realesrgan

API = "/api/v1"
UNIMPLEMENTED = {
    "subtitle.autogenerate", "audio.music.generate", "audio.separate.stems",
    "image.background.remove", "image.inpaint", "image.restore.face", "image.tag",
    "video.upscale",
}
PROVIDER_CAPABILITIES = {"audio.transcribe", "audio.tts", "image.caption"}


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_readiness_enums_and_all_registry_declarations() -> None:
    assert {v.value for v in ImplementationReadiness} == {"available", "unavailable"}
    assert {v.value for v in DeploymentReadiness} == {
        "ready", "configuration_required", "unavailable"
    }
    assert {v.value for v in ReadinessReason} == {
        "not_implemented", "provider_not_configured", "gpu_unavailable",
        "configuration_required", "capability_disabled",
    }
    assert len(registry.CAPABILITIES) == 41
    registry.validate_implementation_registry()
    unavailable = {
        c.id for c in registry.CAPABILITIES
        if c.implementation == ImplementationReadiness.UNAVAILABLE
    }
    assert unavailable == UNIMPLEMENTED
    for cap in registry.CAPABILITIES:
        assert cap.enabled is True
        handler = registry.get_handler(cap.id)
        assert isinstance(handler, NotImplementedHandler) == (cap.id in UNIMPLEMENTED)
        if cap.id in UNIMPLEMENTED:
            assert not resolve_availability(cap).executable
            assert resolve_availability(cap).reason_code == ReadinessReason.NOT_IMPLEMENTED
        else:
            assert cap.implementation == ImplementationReadiness.AVAILABLE
    assert {
        c.id for c in registry.CAPABILITIES
        if c.deployment_requirement == DeploymentRequirement.INFERENCE_PROVIDER
    } == PROVIDER_CAPABILITIES
    assert registry.get("image.upscale").deployment_requirement == DeploymentRequirement.CUDA_WORKER


@pytest.mark.parametrize("direction", ["missing_handler", "unexpected_handler"])
def test_registry_invariants_detect_drift(direction: str, monkeypatch) -> None:
    if direction == "missing_handler":
        monkeypatch.delitem(registry._HANDLERS, "video.trim")
    else:
        monkeypatch.setitem(registry._HANDLERS, "video.upscale", registry.get_handler("video.trim"))
    with pytest.raises(RuntimeError, match="readiness/handler mismatch"):
        registry.validate_implementation_registry()


def test_registry_requires_provider_deployment_declarations(monkeypatch) -> None:
    definitions = [
        cap.model_copy(update={"deployment_requirement": DeploymentRequirement.LOCAL})
        if cap.id == "image.caption" else cap
        for cap in registry.CAPABILITIES
    ]
    monkeypatch.setattr(registry, "CAPABILITIES", definitions)
    with pytest.raises(RuntimeError, match="Provider deployment requirement missing"):
        registry.validate_implementation_registry()


def test_real_handler_cannot_accidentally_be_declared_unimplemented(monkeypatch) -> None:
    definitions = [
        cap.model_copy(update={"implementation": ImplementationReadiness.UNAVAILABLE})
        if cap.id == "video.trim" else cap
        for cap in registry.CAPABILITIES
    ]
    monkeypatch.setattr(registry, "CAPABILITIES", definitions)
    with pytest.raises(RuntimeError, match="readiness/handler mismatch"):
        registry.validate_implementation_registry()


async def test_capability_listing_includes_unavailable_with_additive_fields(client) -> None:
    response = await client.get(f"{API}/capabilities")
    assert response.status_code == 200
    caps = {c["id"]: c for c in response.json()}
    assert len(caps) == 41
    for capability_id in UNIMPLEMENTED:
        cap = caps[capability_id]
        assert cap["enabled"] is True
        assert cap["readiness"] == {
            "implementation": "unavailable", "deployment": "ready",
            "reason_code": "not_implemented",
        }
        assert cap["availability"] == {"executable": False, "reason_code": "not_implemented"}
        assert cap["applicability"] == registry.get(capability_id).applicability.model_dump(
            mode="json"
        )
        assert "x-ui" not in str(cap)
    assert caps["project.export"]["availability"] == {"executable": True, "reason_code": None}
    assert caps["image.upscale"]["readiness"]["implementation"] == "available"
    detail = (await client.get(f"{API}/capabilities/video.upscale")).json()
    assert detail == caps["video.upscale"]
    assert capability_read(registry.get("video.upscale")).availability.executable is False


async def test_disabled_capability_discovery_does_not_enable_job_admission(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    project_id = await make_project("Disabled discovery")
    disabled = registry.get("video.trim").model_copy(update={"enabled": False})
    monkeypatch.setattr(
        registry,
        "CAPABILITIES",
        [disabled if cap.id == disabled.id else cap for cap in registry.CAPABILITIES],
    )
    monkeypatch.setitem(registry.CAPABILITIES_BY_ID, disabled.id, disabled)

    default_list = await client.get(f"{API}/capabilities")
    assert default_list.status_code == 200
    assert disabled.id not in {cap["id"] for cap in default_list.json()}

    explicit_false = await client.get(f"{API}/capabilities?include_disabled=false")
    assert explicit_false.status_code == 200
    assert disabled.id not in {cap["id"] for cap in explicit_false.json()}

    include_disabled = await client.get(f"{API}/capabilities?include_disabled=true")
    assert include_disabled.status_code == 200
    listed = {cap["id"]: cap for cap in include_disabled.json()}
    assert listed[disabled.id]["enabled"] is False
    assert listed[disabled.id]["readiness"]["implementation"] == "available"
    assert listed[disabled.id]["availability"] == {
        "executable": False,
        "reason_code": "capability_disabled",
    }

    detail = await client.get(f"{API}/capabilities/{disabled.id}")
    assert detail.status_code == 200
    assert detail.json() == listed[disabled.id]

    uploaded = await client.post(
        f"{API}/projects/{project_id}/assets",
        files={"file": ("source.mp4", b"not executed", "video/mp4")},
    )
    assert uploaded.status_code == 201
    queued: list[str] = []

    async def enqueue(job_id):
        queued.append(job_id)

    app.dependency_overrides[get_enqueuer] = lambda: enqueue
    job_response = await client.post(
        f"{API}/jobs",
        json={
            "capability_id": disabled.id,
            "project_id": project_id,
            "asset_id": uploaded.json()["id"],
            "input": {"start": 0, "end": 1},
        },
    )
    assert job_response.status_code == 422
    assert job_response.json()["error"]["code"] == "capability_unavailable"
    assert queued == []


@pytest.mark.parametrize("capability_id", sorted(UNIMPLEMENTED))
async def test_unavailable_http_admission_never_persists_or_enqueues(
    client: AsyncClient, make_project, capability_id: str
) -> None:
    pid = await make_project("Readiness")
    cap = registry.get(capability_id)
    body = {"capability_id": capability_id, "project_id": pid, "input": {}}
    if cap.supported_media_types:
        media = cap.supported_media_types[0].value
        suffix, mime = {
            "video": ("mp4", "video/mp4"), "image": ("png", "image/png"),
            "audio": ("wav", "audio/wav"),
        }[media]
        uploaded = await client.post(
            f"{API}/projects/{pid}/assets", files={"file": (f"source.{suffix}", b"source", mime)}
        )
        assert uploaded.status_code == 201
        body["asset_id"] = uploaded.json()["id"]
    else:
        body["input"] = {"prompt": "Test music"}
    queued = []

    async def enqueue(job_id):
        queued.append(job_id)

    app.dependency_overrides[get_enqueuer] = lambda: enqueue
    response = await client.post(f"{API}/jobs", json=body)
    assert response.status_code == 422, response.text
    assert response.json()["error"]["code"] == "capability_unavailable"
    assert "not_implemented" in response.json()["error"]["message"]
    assert queued == []
    async with async_session_factory() as session:
        assert await session.scalar(select(func.count()).select_from(Job)) == 0


async def test_service_admission_is_also_guarded() -> None:
    async with async_session_factory() as session:
        with pytest.raises(CapabilityUnavailableError, match="not_implemented"):
            await create_job(session, JobCreate(
                capability_id="audio.music.generate", project_id=uuid.uuid4(),
                input={"prompt": "Test"},
            ))
        assert await session.scalar(select(func.count()).select_from(Job)) == 0


async def test_fallback_direct_invocation_always_fails() -> None:
    with pytest.raises(CapabilityUnavailableError, match="not_implemented"):
        await NotImplementedHandler().run(JobContext(capability_id="legacy.operation"))


@pytest.mark.parametrize("old_queued", [True, False])
async def test_old_queued_unavailable_and_fallback_worker_jobs_fail_without_outputs(
    client, make_project, monkeypatch, old_queued: bool
) -> None:
    pid = uuid.UUID(await make_project("Fallback safety"))
    capability_id = "image.tag" if old_queued else "media.titlecard"
    if not old_queued:
        # Exercise fallback itself after normal readiness admits a real definition.
        monkeypatch.setitem(registry._HANDLERS, capability_id, NotImplementedHandler())
    async with async_session_factory() as session:
        job = Job(project_id=pid, capability_id=capability_id, input={"text": "Title"})
        session.add(job)
        await session.commit()
        job = await execute_job(session, job.id)
        assert job.status == JobStatus.failed
        assert "not_implemented" in job.error
        assert not job.output
        assert await session.scalar(select(func.count()).select_from(Asset)) == 0


@pytest.mark.parametrize(
    "provider,capability_id,extra,ready",
    [
        ("mock", "audio.transcribe", {}, False),
        ("mock", "audio.tts", {}, False),
        ("mock", "image.caption", {}, False),
        ("http", "image.caption", {}, False),
        ("http", "image.caption", {"inference_base_url": "http://inference.test"}, True),
        ("http", "audio.transcribe", {"inference_base_url": "http://inference.test"}, True),
        ("http", "audio.tts", {"inference_base_url": "http://inference.test"}, False),
        ("http", "image.caption", {
            "inference_base_url": "http://inference.test", "inference_image_caption_path": ""
        }, False),
        ("openrouter", "audio.tts", {"openrouter_api_key": ""}, False),
        ("openrouter", "audio.tts", {"openrouter_api_key": "test-only-key"}, True),
        ("openrouter", "audio.transcribe", {"openrouter_api_key": "test-only-key"}, True),
        ("openrouter", "image.caption", {"openrouter_api_key": "test-only-key"}, False),
        ("openrouter", "audio.tts", {
            "openrouter_api_key": "test-only-key", "openrouter_tts_model": ""
        }, False),
        ("unknown", "audio.transcribe", {}, False),
    ],
)
def test_provider_readiness_matches_configuration_and_supported_operations(
    provider, capability_id, extra, ready
) -> None:
    config = _settings(inference_provider=provider, **extra)
    cap = registry.get(capability_id)
    result = resolve_availability(cap, settings=config)
    assert cap.implementation == ImplementationReadiness.AVAILABLE
    assert result.executable is ready
    assert result.reason_code == (None if ready else ReadinessReason.PROVIDER_NOT_CONFIGURED)


async def test_mock_configuration_does_not_admit_even_source_less_jobs(
    client, make_project
) -> None:
    pid = await make_project("Mock unavailable")
    response = await client.post(f"{API}/jobs", json={
        "capability_id": "audio.tts", "project_id": pid, "input": {"text": "Test"},
    })
    assert response.status_code == 422
    assert "provider_not_configured" in response.json()["error"]["message"]
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0


async def test_provider_runtime_failure_stays_a_failed_job_and_keeps_readiness(
    client, make_project, monkeypatch
) -> None:
    class BrokenProvider(BaseInferenceProvider):
        async def infer(self, request: InferenceRequest) -> dict:
            raise ProviderError("Provider temporarily unavailable")

    monkeypatch.setattr(settings, "inference_provider", "http")
    monkeypatch.setattr(settings, "inference_base_url", "http://inference.test")
    monkeypatch.setattr(handler_base, "get_provider", lambda: BrokenProvider())
    pid = await make_project("Configured provider")
    asset = (await client.post(f"{API}/projects/{pid}/assets", files={
        "file": ("pic.png", b"source", "image/png"),
    })).json()["id"]
    response = await client.post(f"{API}/jobs", json={
        "capability_id": "image.caption", "asset_id": asset, "project_id": pid,
    })
    assert response.status_code == 201
    job = (await client.get(f"{API}/jobs/{response.json()['id']}")).json()
    assert job["status"] == "failed"
    assert "Provider temporarily unavailable" in job["error"]
    assert not job["output"]
    assert resolve_availability(registry.get("image.caption")).executable


async def test_provider_configuration_changed_after_queue_is_rechecked(
    client, make_project, monkeypatch
) -> None:
    pid = uuid.UUID(await make_project("Queued config change"))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-only-key")
    async with async_session_factory() as session:
        job = await create_job(session, JobCreate(
            project_id=pid, capability_id="audio.tts", input={"text": "Test"},
        ))
        monkeypatch.setattr(settings, "inference_provider", "mock")
        result = await execute_job(session, job.id)
        assert result.status == JobStatus.failed
        assert "provider_not_configured" in result.error
        assert not result.output


async def test_tts_missing_audio_is_truthful_provider_failure(monkeypatch) -> None:
    class EmptyProvider(BaseInferenceProvider):
        async def infer(self, request):
            return {"provider": "test"}

    monkeypatch.setattr("app.capabilities.handlers.audio_tts.get_provider", lambda: EmptyProvider())
    with pytest.raises(ProviderError, match="no audio output"):
        await AudioTtsHandler().run(JobContext(capability_id="audio.tts", params={"text": "Test"}))


def test_gpu_readiness_uses_worker_declaration_without_api_inference(monkeypatch) -> None:
    def no_probe():
        raise AssertionError("API readiness must not probe the API container's CUDA device")

    monkeypatch.setattr(realesrgan, "check_runtime_readiness", no_probe)
    cap = registry.get("image.upscale")
    result = resolve_readiness(cap, settings=_settings(image_upscale_worker_ready=False))
    assert result.implementation == ImplementationReadiness.AVAILABLE
    assert result.deployment == DeploymentReadiness.CONFIGURATION_REQUIRED
    assert result.reason_code == ReadinessReason.CONFIGURATION_REQUIRED
    assert resolve_availability(cap, settings=_settings(image_upscale_worker_ready=True)).executable
    assert not resolve_availability(registry.get("video.upscale")).executable


def test_gpu_prerequisite_probe_rejects_cuda_absence_without_cpu_fallback(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    ))
    monkeypatch.setattr(realesrgan, "_upsampler", None)
    with pytest.raises(CapabilityUnavailableError, match="gpu_unavailable"):
        realesrgan.check_runtime_readiness()
    with pytest.raises(ToolError, match="no CPU fallback"):
        realesrgan._get_upsampler()


def test_gpu_prerequisite_probe_checks_model_without_loading_it(monkeypatch, tmp_path) -> None:
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True)
    ))
    monkeypatch.setattr(realesrgan, "MODEL_PATH", str(tmp_path / "absent.pth"))
    with pytest.raises(CapabilityUnavailableError, match="configuration_required"):
        realesrgan.check_runtime_readiness()
    model = tmp_path / "present.pth"
    model.touch()
    monkeypatch.setattr(realesrgan, "MODEL_PATH", str(model))
    realesrgan.check_runtime_readiness()


async def test_gpu_worker_does_not_execute_when_runtime_is_unavailable(
    client, make_project, monkeypatch
) -> None:
    pid = await make_project("GPU runtime")
    aid = (await client.post(f"{API}/projects/{pid}/assets", files={
        "file": ("pic.png", b"source", "image/png"),
    })).json()["id"]
    monkeypatch.setattr(settings, "image_upscale_worker_ready", True)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    ))
    response = await client.post(f"{API}/jobs", json={
        "capability_id": "image.upscale", "project_id": pid, "asset_id": aid,
    })
    assert response.status_code == 201
    result = (await client.get(f"{API}/jobs/{response.json()['id']}")).json()
    assert result["status"] == "failed"
    assert "gpu_unavailable" in result["error"]
    assert not result["output"]


def test_global_enabled_is_a_separate_gate_and_reason() -> None:
    cap = registry.get("video.trim").model_copy(update={"enabled": False})
    assert resolve_readiness(cap).implementation == ImplementationReadiness.AVAILABLE
    assert resolve_availability(cap).reason_code == ReadinessReason.CAPABILITY_DISABLED
    with pytest.raises(CapabilityUnavailableError, match="capability_disabled"):
        require_executable(cap)


def test_agent_catalog_and_validator_use_readiness_generically(monkeypatch) -> None:
    monkeypatch.setattr(settings, "agent_deterministic_only", False)
    ids = {cap["capability_id"] for cap in build_catalog()}
    assert not ids.intersection(UNIMPLEMENTED | PROVIDER_CAPABILITIES | {"image.upscale"})
    assert {"video.trim", "project.export", "media.titlecard"} <= ids
    for capability_id in UNIMPLEMENTED | PROVIDER_CAPABILITIES | {"image.upscale"}:
        plan = Plan.model_validate({"type": "plan", "summary": "Test", "steps": [{
            "id": "step", "capability_id": capability_id, "params": {},
        }]})
        with pytest.raises(PlanValidationError, match="unavailable"):
            validate_plan(plan, {})
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-only-key")
    assert "audio.tts" in {c["capability_id"] for c in build_catalog()}
    plan = Plan.model_validate({"type": "plan", "summary": "TTS", "steps": [{
        "id": "tts", "capability_id": "audio.tts", "params": {"text": "Test"},
    }]})
    validate_plan(plan, {})


async def test_agent_approval_rechecks_readiness_before_creating_attempt(
    client, make_project
) -> None:
    pid = uuid.UUID(await make_project("Stale plan"))
    async with async_session_factory() as session:
        agent_session = AgentSession(project_id=pid)
        session.add(agent_session)
        await session.flush()
        plan = AgentPlan(session_id=agent_session.id, summary="Stale unavailable", steps=[{
            "id": "music", "capability_id": "audio.music.generate", "params": {"prompt": "Test"},
        }])
        session.add(plan)
        await session.commit()
        plan_id = plan.id
    response = await client.post(f"{API}/agent/plans/{plan_id}/approve")
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "capability_unavailable"
    async with async_session_factory() as session:
        plan = await session.get(AgentPlan, plan_id)
        assert plan.status == AgentPlanStatus.proposed
        assert await session.scalar(select(func.count()).select_from(AgentPlanAttempt)) == 0


async def test_agent_claimed_attempt_rechecks_configuration_truthfully(
    client, make_project, monkeypatch
) -> None:
    pid = uuid.UUID(await make_project("Worker plan config change"))
    monkeypatch.setattr(settings, "inference_provider", "openrouter")
    monkeypatch.setattr(settings, "openrouter_api_key", "test-only-key")
    async with async_session_factory() as session:
        agent_session = AgentSession(project_id=pid)
        session.add(agent_session)
        await session.flush()
        plan = AgentPlan(session_id=agent_session.id, summary="TTS", steps=[{
            "id": "tts", "capability_id": "audio.tts", "params": {"text": "Test"},
        }])
        session.add(plan)
        await session.commit()
        plan, attempt, *_ = await agent_svc.approve_plan(session, plan.id)
        monkeypatch.setattr(settings, "inference_provider", "mock")
        result = await execute_plan(session, plan.id, attempt.id)
        assert result.status == AgentPlanStatus.failed
        assert "provider_not_configured" in result.error
        assert attempt.status == "failed"
        assert await session.scalar(select(func.count()).select_from(Job)) == 0


def test_openapi_exposes_stable_readiness_and_availability_models() -> None:
    schemas = app.openapi()["components"]["schemas"]
    assert "readiness" in schemas["CapabilityRead"]["required"]
    assert "availability" in schemas["CapabilityRead"]["required"]
    assert schemas["ImplementationReadiness"]["enum"] == ["available", "unavailable"]
    assert schemas["CapabilityAvailability"]["properties"]["executable"]["type"] == "boolean"
    assert "deployment_requirement" not in schemas["CapabilityRead"]["properties"]
    assert "x-ui" not in str(schemas["CapabilityRead"])
