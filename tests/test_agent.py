"""Agent tests: catalog, plan validation, mock planner, and session flow.

Planning/validation tests need no binaries or network (AGENT_PROVIDER=mock). The
end-to-end execution test is skip-guarded on ffmpeg/convert/font.
"""

import io
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from app.agent.catalog import build_catalog, catalog_text
from app.agent.llm import SYSTEM_INSTRUCTIONS
from app.agent.planner import PlanValidationError, validate_plan
from app.agent.render import compile_execution_plan, render_group
from app.agent.schemas import Plan
from app.capabilities import registry
from app.core.database import async_session_factory
from app.core.errors import ValidationError
from app.models.agent import AgentPlan
from app.models.asset import Asset
from app.models.job import Job
from app.tools.imagemagick import DEFAULT_FONT

API = "/api/v1"
NEED_ALL = (
    shutil.which("ffmpeg") and shutil.which("convert") and os.path.exists(DEFAULT_FONT)
)


# --- catalog (no binaries) --------------------------------------------------


def test_catalog_reflects_registry() -> None:
    ids = {c["capability_id"] for c in build_catalog()}
    assert {"video.trim", "image.slideshow", "video.compose"} <= ids
    for c in build_catalog():
        assert "input_schema" in c and "supported_media_types" in c
        assert c["applicability"] == registry.get(c["capability_id"]).applicability.model_dump(
            mode="json"
        )
    project_export = next(
        c for c in build_catalog() if c["capability_id"] == "project.export"
    )
    assert project_export["applicability"] == {
        "primary_asset": "forbidden",
        "project_context": "required",
    }


def test_catalog_text_exposes_compact_contract_without_ui_hints() -> None:
    text = catalog_text()
    mix_line = next(line for line in text.splitlines() if line.startswith("- audio.mix "))
    assert "music_asset_id: string/uuid required" in mix_line
    assert "asset_role=background_audio" in mix_line
    assert "asset_media=['audio']" in mix_line
    assert "music_volume: number optional default=0.3 unit=ratio" in mix_line
    assert "enum=[" in mix_line and "mix (Mix supplied audio" in mix_line
    assert "x-ui" not in text
    assert "primary_asset=required" in text
    assert "project_context=required" in text
    transcode_line = next(
        line for line in text.splitlines() if line.startswith("- video.transcode ")
    )
    assert "libx264 (H.264 / AVC)" in transcode_line
    assert "choices_when_format=" in transcode_line
    assert "webm" in transcode_line and "libvpx-vp9" in transcode_line
    assert "video.trim keeps" in SYSTEM_INSTRUCTIONS
    assert "video.split divides the complete input" in SYSTEM_INSTRUCTIONS
    assert "ask a clarification question" in SYSTEM_INSTRUCTIONS
    trim = next(c for c in build_catalog() if c["capability_id"] == "video.trim")
    split = next(c for c in build_catalog() if c["capability_id"] == "video.split")
    assert "Keep one specified interval" in trim["description"]
    assert "does not choose which segment to keep" in split["description"]

    normalize_line = next(
        line for line in text.splitlines() if line.startswith("- audio.normalize ")
    )
    assert "video/audio" in normalize_line
    assert "video input preserves the video stream" in normalize_line
    extract_line = next(
        line for line in text.splitlines() if line.startswith("- video.audio.extract ")
    )
    assert "separate AAC/M4A audio asset" in extract_line
    assert "does not return video" in extract_line

    concat_line = next(
        line for line in text.splitlines() if line.startswith("- video.concat ")
    )
    assert "asset_ids: array<string/uuid> required min_items=2" in concat_line


# --- plan validation (no binaries) -----------------------------------------

IMG = "11111111-1111-1111-1111-111111111111"
VID = "22222222-2222-2222-2222-222222222222"
SCOPE = {IMG: "image", VID: "video"}


def _plan(steps: list[dict]) -> Plan:
    return Plan.model_validate({"type": "plan", "steps": steps})


def test_valid_plan_passes() -> None:
    plan = _plan([
        {"id": "s1", "capability_id": "image.resize",
         "params": {"width": 64, "height": 48}, "asset": IMG},
    ])
    validate_plan(plan, SCOPE)  # no raise


def test_planner_enforces_primary_asset_applicability_generically() -> None:
    missing_primary = _plan(
        [{"id": "s1", "capability_id": "video.trim", "params": {"start": 0, "end": 1}}]
    )
    with pytest.raises(PlanValidationError, match="requires a primary asset"):
        validate_plan(missing_primary, SCOPE)

    forbidden_primary = _plan(
        [{"id": "s1", "capability_id": "media.titlecard", "asset": VID,
          "params": {"text": "Intro"}}]
    )
    with pytest.raises(PlanValidationError, match="does not accept a primary asset"):
        validate_plan(forbidden_primary, SCOPE)

    source_less = _plan(
        [{"id": "s1", "capability_id": "media.titlecard", "params": {"text": "Intro"}}]
    )
    validate_plan(source_less, SCOPE)

    multi_input = _plan(
        [{"id": "s1", "capability_id": "video.compose", "assets": [VID],
          "params": {}}]
    )
    validate_plan(multi_input, SCOPE)


def test_unknown_capability_rejected() -> None:
    plan = _plan([{"id": "s1", "capability_id": "does.not.exist", "params": {}}])
    with pytest.raises(PlanValidationError):
        validate_plan(plan, SCOPE)


def test_bad_params_rejected() -> None:
    # image.resize requires width+height
    plan = _plan([{"id": "s1", "capability_id": "image.resize",
                   "params": {"width": 64}, "asset": IMG}])
    with pytest.raises(PlanValidationError):
        validate_plan(plan, SCOPE)


@pytest.mark.parametrize(
    "capability_id,params",
    [
        ("image.resize", {"width": "wide", "height": 48}),
        ("image.format", {"format": "bmp"}),
        ("video.concat", {"asset_ids": [VID]}),
        ("video.concat", {"asset_ids": ["not-a-uuid", VID]}),
        ("video.transcode", {"format": "webm", "video_codec": "libx264"}),
    ],
)
def test_planner_rejects_same_malformed_contract_as_direct_jobs(
    capability_id: str, params: dict
) -> None:
    plan = _plan([{"id": "s1", "capability_id": capability_id, "params": params}])
    with pytest.raises(PlanValidationError):
        validate_plan(plan, SCOPE)


def test_planner_rejects_wrong_media_for_named_and_list_asset_parameters() -> None:
    audio = "33333333-3333-3333-3333-333333333333"
    scope = {**SCOPE, audio: "audio"}
    wrong_music = _plan(
        [
            {
                "id": "s1",
                "capability_id": "audio.mix",
                "asset": VID,
                "params": {"music_asset_id": IMG},
            }
        ]
    )
    with pytest.raises(PlanValidationError):
        validate_plan(wrong_music, scope)

    wrong_segment = _plan(
        [
            {
                "id": "s1",
                "capability_id": "video.concat",
                "params": {"asset_ids": [VID, IMG]},
            }
        ]
    )
    with pytest.raises(PlanValidationError):
        validate_plan(wrong_segment, scope)

    valid_music = _plan(
        [
            {
                "id": "s1",
                "capability_id": "audio.mix",
                "asset": VID,
                "params": {"music_asset_id": audio},
            }
        ]
    )
    validate_plan(valid_music, scope)


def test_media_type_mismatch_rejected() -> None:
    # video.trim against an image primary asset
    plan = _plan([{"id": "s1", "capability_id": "video.trim",
                   "params": {"start": 0, "end": 1}, "asset": IMG}])
    with pytest.raises(PlanValidationError):
        validate_plan(plan, SCOPE)


def test_bad_step_ref_rejected() -> None:
    plan = _plan([{"id": "s1", "capability_id": "video.compose",
                   "params": {}, "assets": ["@missing"]}])
    with pytest.raises(PlanValidationError):
        validate_plan(plan, SCOPE)


def test_split_output_cannot_be_referenced_as_one_asset() -> None:
    plan = _plan([
        {"id": "split", "capability_id": "video.split", "params": {"markers": [2]},
         "asset": VID},
        {"id": "resize", "capability_id": "video.resize",
         "params": {"width": 80, "height": 60}, "asset": "@split"},
    ])
    with pytest.raises(PlanValidationError, match="returns multiple segments"):
        validate_plan(plan, SCOPE)


def test_render_compiler_groups_chain_and_counts_encoded_streams() -> None:
    plan = _plan([
        {"id": "trim", "capability_id": "video.trim", "asset": VID,
         "params": {"start": 1, "end": 5}},
        {"id": "resize", "capability_id": "video.resize", "asset": "@trim",
         "params": {"width": 640, "height": 360}},
        {"id": "lower", "capability_id": "video.lowerthird", "asset": "@resize",
         "params": {"text": "Name", "start": 0.5, "duration": 2}},
    ])
    facts = {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "a" * 64,
        "technical_metadata_status": "available", "width": 1920, "height": 1080,
        "has_audio_stream": True,
    }}
    summary = compile_execution_plan(plan, facts)
    assert summary["render_group_count"] == 1
    assert summary["video_encode_count"] == 1
    assert summary["audio_encode_count"] == 1
    assert summary["encode_counts_complete"] is True
    assert [node["kind"] for node in summary["nodes"]] == ["render_group"]
    group = summary["nodes"][0]
    assert group["step_ids"] == ["trim", "resize", "lower"]
    assert [op["operation"]["kind"] for op in group["operations"]] == [
        "temporaltrim", "resize", "lowerthird"
    ]

    resize_only = _plan([{
        "id": "resize", "capability_id": "video.resize", "asset": VID,
        "params": {"width": 1280, "height": 720},
    }])
    resize_summary = compile_execution_plan(resize_only, {
        VID: {**facts[VID], "has_audio_stream": True}
    })
    assert resize_summary["render_group_count"] == 1
    assert resize_summary["video_encode_count"] == 1
    assert resize_summary["audio_encode_count"] == 0


def test_render_compiler_preserves_order_and_eliminates_only_proven_identity() -> None:
    crop_resize = _plan([
        {"id": "crop", "capability_id": "video.crop", "asset": VID,
         "params": {"x": 5, "y": 5, "width": 90, "height": 70}},
        {"id": "resize", "capability_id": "video.resize", "asset": "@crop",
         "params": {"width": 50, "height": 40}},
    ])
    resize_crop = _plan([
        {"id": "resize", "capability_id": "video.resize", "asset": VID,
         "params": {"width": 100, "height": 80}},
        {"id": "crop", "capability_id": "video.crop", "asset": "@resize",
         "params": {"x": 0, "y": 0, "width": 50, "height": 40}},
    ])
    facts = {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "b" * 64,
        "technical_metadata_status": "available", "width": 100, "height": 80,
        "has_audio_stream": False,
    }}
    first = compile_execution_plan(crop_resize, facts)["nodes"][0]
    second = compile_execution_plan(resize_crop, facts)["nodes"][0]
    assert [op["operation"]["kind"] for op in first["operations"]] == ["crop", "resize"]
    assert first["operations"][0]["identity"] is False
    assert [op["operation"]["kind"] for op in second["operations"]] == ["resize", "crop"]
    assert second["operations"][1]["identity"] is False

    resized_crop = _plan([
        {"id": "resize", "capability_id": "video.resize", "asset": VID,
         "params": {"width": 50, "height": 40}},
        {"id": "crop", "capability_id": "video.crop", "asset": "@resize",
         "params": {"x": 0, "y": 0, "width": 50, "height": 40}},
    ])
    propagated = compile_execution_plan(resized_crop, facts)["nodes"][0]
    assert propagated["operations"][1]["identity"] is True


@pytest.mark.parametrize(
    "boundary", ["video.upscale", "video.split", "video.audio.extract", "video.concat"]
)
def test_render_group_closes_at_unsupported_operation_boundaries(boundary: str) -> None:
    params = {"markers": [2]} if boundary == "video.split" else {}
    if boundary == "video.concat":
        params = {"asset_ids": [VID, "another-asset"]}
    plan = _plan([
        {"id": "trim", "capability_id": "video.trim", "asset": VID,
         "params": {"start": 1, "end": 5}},
        {"id": "boundary", "capability_id": boundary, "asset": "@trim", "params": params},
        {"id": "resize", "capability_id": "video.resize", "asset": "@boundary",
         "params": {"width": 640, "height": 360}},
    ])
    summary = compile_execution_plan(plan, {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "a" * 64,
        "technical_metadata_status": "available", "width": 1920, "height": 1080,
        "has_audio_stream": True,
    }})
    assert [node["kind"] for node in summary["nodes"]] == ["render_group", "job", "job"]


def test_render_group_does_not_absorb_step_with_extra_dependency() -> None:
    plan = _plan([
        {"id": "trim", "capability_id": "video.trim", "asset": VID,
         "params": {"start": 1, "end": 5}},
        {"id": "resize", "capability_id": "video.resize", "asset": "@trim",
         "depends_on": ["other"], "params": {"width": 640, "height": 360}},
    ])
    summary = compile_execution_plan(plan, {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "a" * 64,
        "technical_metadata_status": "available", "width": 1920, "height": 1080,
        "has_audio_stream": True,
    }})
    assert [node["kind"] for node in summary["nodes"]] == ["render_group", "job"]


def test_identity_only_crop_has_no_render_job_and_stays_auditable() -> None:
    plan = _plan([{
        "id": "crop", "capability_id": "video.crop", "asset": VID,
        "params": {"x": 0, "y": 0, "width": 100, "height": 80},
    }])
    facts = {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "e" * 64,
        "technical_metadata_status": "available", "width": 100, "height": 80,
        "has_audio_stream": False,
    }}
    summary = compile_execution_plan(plan, facts)
    assert summary["nodes"][0]["kind"] == "identity"
    assert summary["render_group_count"] == 0
    assert summary["steps"][0]["identity"] is True


def test_unknown_dimensions_keep_crop_and_unsupported_step_breaks_group() -> None:
    plan = _plan([
        {"id": "crop", "capability_id": "video.crop", "asset": VID,
         "params": {"x": 0, "y": 0, "width": 100, "height": 80}},
        {"id": "mute", "capability_id": "video.mute", "asset": "@crop", "params": {}},
        {"id": "resize", "capability_id": "video.resize", "asset": "@mute",
         "params": {"width": 50, "height": 40}},
    ])
    facts = {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "c" * 64,
        "technical_metadata_status": "pending", "width": None, "height": None,
        "has_audio_stream": None,
    }}
    summary = compile_execution_plan(plan, facts)
    assert [node["kind"] for node in summary["nodes"]] == ["render_group", "job", "job"]
    assert summary["nodes"][0]["operations"][0]["identity"] is False
    assert summary["video_encode_count"] is None
    assert summary["encode_counts_complete"] is False


@pytest.mark.asyncio
async def test_render_filtergraph_preserves_logical_order(monkeypatch, tmp_path) -> None:
    from app.tools import ffmpeg, imagemagick, runner

    calls: list[list[str]] = []

    async def no_audio(_path: str) -> bool:
        return False

    async def capture(args: list[str], **_kwargs) -> str:
        calls.append(args)
        return ""

    async def make_overlay(dst: str, *_args, **_kwargs) -> None:
        from pathlib import Path
        Path(dst).write_bytes(b"png")

    monkeypatch.setattr(ffmpeg, "_has_audio", no_audio)
    monkeypatch.setattr(runner, "run", capture)
    monkeypatch.setattr(imagemagick, "render_lowerthird", make_overlay)
    plan = _plan([
        {"id": "trim", "capability_id": "video.trim", "asset": VID,
         "params": {"start": 10, "end": 20}},
        {"id": "trim2", "capability_id": "video.trim", "asset": "@trim",
         "params": {"start": 2, "end": 5}},
        {"id": "crop", "capability_id": "video.crop", "asset": "@trim2",
         "params": {"x": 10, "y": 5, "width": 90, "height": 70}},
        {"id": "resize", "capability_id": "video.resize", "asset": "@crop",
         "params": {"width": 80, "height": 60}},
        {"id": "lower", "capability_id": "video.lowerthird", "asset": "@resize",
         "params": {"text": "Name", "start": 2, "duration": 3}},
    ])
    group = compile_execution_plan(plan, {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "d" * 64,
        "technical_metadata_status": "pending", "has_audio_stream": None,
    }})["nodes"][0]
    await render_group(tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path, group)
    assert len(calls) == 1
    graph = calls[0][calls[0].index("-filter_complex") + 1]
    assert graph.index("trim=start=10:end=20") < graph.index("trim=start=2:end=5")
    assert graph.index("trim=start=2:end=5") < graph.index("crop=90:70:10:5")
    assert graph.index("crop=90:70:10:5") < graph.index("scale=80:60")
    assert graph.index("scale=80:60") < graph.index("overlay=")
    assert "enable='between(t,2,5)'" in graph

    overlay_crop = _plan([
        {"id": "lower", "capability_id": "video.lowerthird", "asset": VID,
         "params": {"text": "Name", "start": 0, "duration": 2}},
        {"id": "crop", "capability_id": "video.crop", "asset": "@lower",
         "params": {"x": 10, "y": 5, "width": 90, "height": 70}},
    ])
    reverse_group = compile_execution_plan(overlay_crop, {VID: {
        "media_type": "video", "extension": "mp4", "checksum_sha256": "d" * 64,
        "technical_metadata_status": "pending", "has_audio_stream": None,
    }})["nodes"][0]
    await render_group(tmp_path / "in.mp4", tmp_path / "out.mp4", tmp_path, reverse_group)
    reverse_graph = calls[1][calls[1].index("-filter_complex") + 1]
    assert reverse_graph.index("overlay=") < reverse_graph.index("crop=90:70:10:5")


@pytest.mark.skipif(not NEED_ALL, reason="needs ffmpeg + convert + font")
async def test_agent_trim_resize_lowerthird_executes_one_physical_render(
    client: AsyncClient, make_project, monkeypatch, tmp_path
) -> None:
    from app.agent.schemas import Plan
    from app.models.enums import TechnicalMetadataStatus

    pid = await make_project("agent-render-group")
    video_path = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=4",
        "-f", "lavfi", "-i", "sine=frequency=880:duration=4",
        "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", str(video_path),
    ], check=True)
    with video_path.open("rb") as fh:
        upload = await client.post(
            f"{API}/projects/{pid}/assets",
            files={"file": ("source.mp4", fh.read(), "video/mp4")},
        )
    assert upload.status_code == 201
    asset_id = upload.json()["id"]
    async with async_session_factory() as db:
        asset = await db.get(Asset, uuid.UUID(asset_id))
        assert asset is not None
        asset.technical_metadata_status = TechnicalMetadataStatus.available.value
        asset.width, asset.height, asset.has_audio_stream = 160, 120, True
        await db.commit()

    session_resp = await client.post(
        f"{API}/agent/sessions", json={"project_id": pid, "asset_ids": [asset_id]}
    )
    session_id = session_resp.json()["id"]
    plan = Plan.model_validate({"type": "plan", "summary": "Trim, resize, identify",
        "steps": [
            {"id": "trim", "capability_id": "video.trim", "asset": asset_id,
             "params": {"start": 0.5, "end": 2.5}},
            {"id": "resize", "capability_id": "video.resize", "asset": "@trim",
             "params": {"width": 80, "height": 60}},
            {"id": "lower", "capability_id": "video.lowerthird", "asset": "@resize",
             "params": {"text": "Test name", "start": 0.25, "duration": 0.5}},
        ]})

    async def fake_propose(*_args, **_kwargs):
        return plan, []

    monkeypatch.setattr("app.services.agent.llm.propose", fake_propose)
    proposed = await client.post(
        f"{API}/agent/sessions/{session_id}/messages", json={"content": "Apply edits"}
    )
    assert proposed.status_code == 200, proposed.text
    plan_body = proposed.json()["plan"]
    summary = plan_body["execution_summary"]
    assert summary["render_group_count"] == 1
    assert summary["video_encode_count"] == 1
    assert summary["audio_encode_count"] == 1
    assert summary["encode_counts_complete"] is True

    approved = await client.post(f"{API}/agent/plans/{plan_body['id']}/approve")
    assert approved.status_code == 200
    final = (await client.get(f"{API}/agent/plans/{plan_body['id']}" )).json()
    assert final["status"] == "succeeded", final.get("error")
    runs = final["step_runs"]
    assert len(runs) == 3
    assert len({run["job_id"] for run in runs}) == 1
    assert {run["render_group_id"] for run in runs} == {"render-group-1"}
    assert sum(bool(run.get("output_asset_id")) for run in runs) == 1
    output_id = final["result_asset_ids"][0]
    output = (await client.get(f"{API}/assets/{output_id}")).json()
    assert output["media_type"] == "video"
    assert output["width"] == 80 and output["height"] == 60
    assert output["technical_metadata_status"] == "available"
    assert output["is_intermediate"] is False
    assert output["temp_expires_at"] is None
    async with async_session_factory() as db:
        jobs = list(
            (await db.execute(select(Job).where(Job.project_id == uuid.UUID(pid)))).scalars()
        )
        assets = list(
            (await db.execute(select(Asset).where(Asset.project_id == uuid.UUID(pid)))).scalars()
        )
        plan_row = await db.get(AgentPlan, uuid.UUID(plan_body["id"]))
        assert len(jobs) == 1 and jobs[0].capability_id == "agent.render_group"
        assert len(assets) == 2
        assert plan_row is not None
        group = plan_row.execution_summary["nodes"][0]
        assert group["status"] == "succeeded"
        assert group["job_id"] == str(jobs[0].id)
        assert group["output_asset_id"] == output_id


def test_out_of_scope_asset_rejected() -> None:
    plan = _plan([{"id": "s1", "capability_id": "image.resize",
                   "params": {"width": 64, "height": 48},
                   "asset": "99999999-9999-9999-9999-999999999999"}])
    with pytest.raises(PlanValidationError):
        validate_plan(plan, SCOPE)


async def test_render_group_source_fingerprint_rejects_stale_input(
    client: AsyncClient, make_project
) -> None:
    from app.agent.executor import _validate_render_sources
    from app.models.enums import TechnicalMetadataStatus

    pid = await make_project("agent-stale-input")
    upload = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("source.mp4", b"not-a-real-video", "video/mp4")},
    )
    asset_id = upload.json()["id"]
    async with async_session_factory() as db:
        asset = await db.get(Asset, uuid.UUID(asset_id))
        assert asset is not None
        asset.technical_metadata_status = TechnicalMetadataStatus.available.value
        asset.width, asset.height, asset.has_audio_stream = 160, 120, False
        facts = {asset_id: {
            "media_type": "video", "extension": "mp4",
            "checksum_sha256": asset.checksum_sha256,
            "technical_metadata_status": asset.technical_metadata_status,
            "width": asset.width, "height": asset.height,
            "has_audio_stream": asset.has_audio_stream,
        }}
        plan = _plan([{
            "id": "resize", "capability_id": "video.resize", "asset": asset_id,
            "params": {"width": 80, "height": 60},
        }])
        summary = compile_execution_plan(plan, facts)
        asset.checksum_sha256 = "f" * 64
        await db.commit()
        with pytest.raises(ValidationError, match="source Asset changed"):
            await _validate_render_sources(db, summary, uuid.UUID(pid))


async def test_identity_elimination_rejects_stale_source_metadata(
    client: AsyncClient, make_project
) -> None:
    from app.agent.executor import _validate_render_sources
    from app.models.enums import TechnicalMetadataStatus

    pid = await make_project("agent-stale-identity")
    upload = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("source.mp4", b"not-a-real-video", "video/mp4")},
    )
    asset_id = upload.json()["id"]
    async with async_session_factory() as db:
        asset = await db.get(Asset, uuid.UUID(asset_id))
        assert asset is not None
        asset.technical_metadata_status = TechnicalMetadataStatus.available.value
        asset.width, asset.height = 100, 80
        facts = {asset_id: {
            "media_type": "video", "extension": "mp4",
            "checksum_sha256": asset.checksum_sha256,
            "technical_metadata_status": asset.technical_metadata_status,
            "width": asset.width, "height": asset.height,
        }}
        plan = _plan([{
            "id": "crop", "capability_id": "video.crop", "asset": asset_id,
            "params": {"x": 0, "y": 0, "width": 100, "height": 80},
        }])
        summary = compile_execution_plan(plan, facts)
        assert summary["nodes"][0]["kind"] == "identity"
        asset.height = 90
        await db.commit()
        with pytest.raises(ValidationError, match="metadata changed"):
            await _validate_render_sources(db, summary, uuid.UUID(pid))


async def test_grouped_render_failure_has_truthful_job_and_no_output_asset(
    client: AsyncClient, make_project, monkeypatch
) -> None:
    from app.agent.schemas import Plan

    pid = await make_project("agent-render-failure")
    upload = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": ("source.mp4", b"fixture", "video/mp4")},
    )
    asset_id = upload.json()["id"]
    session_resp = await client.post(
        f"{API}/agent/sessions", json={"project_id": pid, "asset_ids": [asset_id]}
    )
    session_id = session_resp.json()["id"]
    plan = Plan.model_validate({"type": "plan", "summary": "failing render", "steps": [
        {"id": "resize", "capability_id": "video.resize", "asset": asset_id,
         "params": {"width": 80, "height": 60}},
    ]})

    async def fake_propose(*_args, **_kwargs):
        return plan, []

    async def fail_render(*_args, **_kwargs):
        raise RuntimeError("deterministic render failure")

    monkeypatch.setattr("app.services.agent.llm.propose", fake_propose)
    monkeypatch.setattr("app.agent.render.render_group", fail_render)
    proposed = await client.post(
        f"{API}/agent/sessions/{session_id}/messages", json={"content": "Resize"}
    )
    plan_id = proposed.json()["plan"]["id"]
    await client.post(f"{API}/agent/plans/{plan_id}/approve")
    final = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    assert final["status"] == "failed"
    assert final["step_runs"][0]["status"] == "failed"
    assert final["step_runs"][0].get("output_asset_id") is None
    async with async_session_factory() as db:
        jobs = list(
            (await db.execute(select(Job).where(Job.project_id == uuid.UUID(pid)))).scalars()
        )
        assets = list(
            (await db.execute(select(Asset).where(Asset.project_id == uuid.UUID(pid)))).scalars()
        )
        assert len(jobs) == 1 and jobs[0].status.value == "failed"
        assert len(assets) == 1
        assert final["execution_summary"]["nodes"][0]["status"] == "failed"


@pytest.mark.skipif(not NEED_ALL, reason="needs ffmpeg + convert + font")
async def test_agent_checkpoint_is_hidden_immediately_when_later_step_fails(
    client: AsyncClient, make_project, monkeypatch, tmp_path
) -> None:
    from app.agent.schemas import Plan
    from app.models.enums import TechnicalMetadataStatus
    from app.services import jobs as jobs_svc

    pid = await make_project("agent-temp-checkpoint-failure")
    source = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
        "color=c=blue:size=160x120:rate=10:duration=1", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", str(source),
    ], check=True)
    with source.open("rb") as fh:
        uploaded = await client.post(
            f"{API}/projects/{pid}/assets",
            files={"file": ("source.mp4", fh.read(), "video/mp4")},
        )
    source_id = uploaded.json()["id"]
    async with async_session_factory() as db:
        asset = await db.get(Asset, uuid.UUID(source_id))
        assert asset is not None
        asset.technical_metadata_status = TechnicalMetadataStatus.available.value
        asset.width, asset.height, asset.has_audio_stream = 160, 120, False
        await db.commit()

    agent_session = (await client.post(
        f"{API}/agent/sessions", json={"project_id": pid, "asset_ids": [source_id]}
    )).json()
    plan = Plan.model_validate({"type": "plan", "summary": "checkpoint then fail", "steps": [
        {"id": "resize", "capability_id": "video.resize", "asset": source_id,
         "params": {"width": 80, "height": 60}},
        {"id": "normalize", "capability_id": "audio.normalize", "asset": "@resize",
         "params": {}},
    ]})

    async def fake_propose(*_args, **_kwargs):
        return plan, []

    async def copy_render(src: Path, dst: Path, *_args, **_kwargs) -> bool:
        shutil.copyfile(src, dst)
        return False

    class FailingHandler:
        async def run(self, _ctx):
            raise RuntimeError("expected downstream test failure")

    original_get_handler = jobs_svc.registry.get_handler

    def get_handler(capability_id: str):
        if capability_id == "audio.normalize":
            return FailingHandler()
        return original_get_handler(capability_id)

    monkeypatch.setattr("app.services.agent.llm.propose", fake_propose)
    monkeypatch.setattr("app.agent.render.render_group", copy_render)
    monkeypatch.setattr(jobs_svc.registry, "get_handler", get_handler)
    proposed = await client.post(
        f"{API}/agent/sessions/{agent_session['id']}/messages",
        json={"content": "Resize then normalize"},
    )
    plan_id = proposed.json()["plan"]["id"]
    await client.post(f"{API}/agent/plans/{plan_id}/approve")
    result = (await client.get(f"{API}/agent/plans/{plan_id}")).json()
    assert result["status"] == "failed"
    checkpoint_id = result["step_runs"][0]["output_asset_id"]
    checkpoint = (await client.get(f"{API}/assets/{checkpoint_id}")).json()
    assert checkpoint["is_intermediate"] is True
    assert checkpoint["temp_expires_at"] is not None
    default_assets = (await client.get(f"{API}/projects/{pid}/assets")).json()
    included_assets = (await client.get(
        f"{API}/projects/{pid}/assets?include_intermediate=true"
    )).json()
    assert checkpoint_id not in {asset["id"] for asset in default_assets}
    assert checkpoint_id in {asset["id"] for asset in included_assets}


# --- session flow via API (no binaries) ------------------------------------


async def _project_with_image(client: AsyncClient, make_project) -> tuple[str, str]:
    pid = await make_project("agent")
    files = {"file": ("p.png", io.BytesIO(b"img"), "image/png")}
    aid = (await client.post(f"{API}/projects/{pid}/assets", files=files)).json()["id"]
    return pid, aid


async def test_session_clarification_when_params_missing(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("agent")
    files = {"file": ("v.mp4", io.BytesIO(b"vid"), "video/mp4")}
    vid = (await client.post(f"{API}/projects/{pid}/assets", files=files)).json()["id"]

    sess = (await client.post(f"{API}/agent/sessions",
            json={"project_id": pid, "asset_ids": [vid]})).json()
    resp = await client.post(
        f"{API}/agent/sessions/{sess['id']}/messages",
        json={"content": "Trim this video"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "clarification"
    assert "start" in body["missing"]


async def test_session_proposes_plan(client: AsyncClient, make_project) -> None:
    pid, img = await _project_with_image(client, make_project)
    sess = (await client.post(f"{API}/agent/sessions",
            json={"project_id": pid, "asset_ids": [img]})).json()
    resp = await client.post(
        f"{API}/agent/sessions/{sess['id']}/messages",
        json={"content": "Make a slideshow, 4 seconds each, with a title card "
                         "'Museum Highlights' at the start, export as 1080p mp4"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["type"] == "plan"
    cap_ids = [s["capability_id"] for s in body["plan"]["steps"]]
    assert "image.slideshow" in cap_ids and "video.compose" in cap_ids
    assert body["plan"]["status"] == "proposed"
    # transcript persisted
    sdetail = (await client.get(f"{API}/agent/sessions/{sess['id']}")).json()
    assert len(sdetail["transcript"]) >= 2


# --- end-to-end execution (needs binaries) ---------------------------------


@pytest.mark.skipif(not NEED_ALL, reason="needs ffmpeg + convert + font")
async def test_agent_end_to_end_slideshow(
    client: AsyncClient, make_project, tmp_path
) -> None:
    import subprocess

    pid = await make_project("agent-e2e")
    img_ids = []
    for i, color in enumerate(["red", "green"]):
        p = tmp_path / f"i{i}.png"
        subprocess.run(["convert", "-size", "160x120", f"xc:{color}", str(p)], check=True)
        with open(p, "rb") as fh:
            files = {"file": (f"i{i}.png", fh.read(), "image/png")}
        img_ids.append(
            (await client.post(f"{API}/projects/{pid}/assets", files=files)).json()["id"]
        )

    sess = (await client.post(f"{API}/agent/sessions",
            json={"project_id": pid, "asset_ids": img_ids})).json()
    plan = (await client.post(
        f"{API}/agent/sessions/{sess['id']}/messages",
        json={"content": "Make a slideshow, 1 second each, with a title card "
                         "'Highlights' at the start, export as 720p mp4"},
    )).json()["plan"]

    # approve -> eager execution in tests
    approved = (await client.post(f"{API}/agent/plans/{plan['id']}/approve")).json()
    final = (await client.get(f"{API}/agent/plans/{approved['id']}")).json()
    assert final["status"] == "succeeded", final.get("error")
    assert final["result_asset_ids"]

    # one deliverable (the compose output); title card + slideshow are intermediate
    assert len(final["result_asset_ids"]) == 1
    out_id = final["result_asset_ids"][0]
    meta = (await client.get(f"{API}/assets/{out_id}")).json()
    assert meta["media_type"] == "video"
    assert meta["is_intermediate"] is False
    dl = await client.get(f"{API}/assets/{out_id}/download")
    assert dl.status_code == 200 and len(dl.content) > 0

    # the consumed step outputs (title, slides) are marked intermediate
    consumed = {r["output_asset_id"] for r in final["step_runs"]
                if r["step_id"] in ("title", "slides")}
    for aid in consumed:
        temp_asset = (await client.get(f"{API}/assets/{aid}")).json()
        assert temp_asset["is_intermediate"] is True
        assert temp_asset["temp_expires_at"] is not None

    # Temp Processed Assets are excluded by default and included explicitly.
    pid = (await client.get(f"{API}/agent/sessions/{sess['id']}")).json()["project_id"]
    default_list = (await client.get(f"{API}/projects/{pid}/assets")).json()
    all_assets = (
        await client.get(f"{API}/projects/{pid}/assets?include_intermediate=true")
    ).json()
    all_ids = {a["id"] for a in all_assets}
    default_ids = {a["id"] for a in default_list}
    assert consumed <= all_ids
    assert consumed.isdisjoint(default_ids)
    assert out_id in default_ids
