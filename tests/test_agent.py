"""Agent tests: catalog, plan validation, mock planner, and session flow.

Planning/validation tests need no binaries or network (AGENT_PROVIDER=mock). The
end-to-end execution test is skip-guarded on ffmpeg/convert/font.
"""

import io
import os
import shutil

import pytest
from httpx import AsyncClient

from app.agent.catalog import build_catalog
from app.agent.planner import PlanValidationError, validate_plan
from app.agent.schemas import Plan
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


def test_out_of_scope_asset_rejected() -> None:
    plan = _plan([{"id": "s1", "capability_id": "image.resize",
                   "params": {"width": 64, "height": 48},
                   "asset": "99999999-9999-9999-9999-999999999999"}])
    with pytest.raises(PlanValidationError):
        validate_plan(plan, SCOPE)


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
        assert (await client.get(f"{API}/assets/{aid}")).json()["is_intermediate"] is True

    # the default list shows everything; filtered list hides intermediates
    pid = (await client.get(f"{API}/agent/sessions/{sess['id']}")).json()["project_id"]
    all_assets = (await client.get(f"{API}/projects/{pid}/assets")).json()
    finals_only = (
        await client.get(f"{API}/projects/{pid}/assets?include_intermediate=false")
    ).json()
    all_ids = {a["id"] for a in all_assets}
    final_ids = {a["id"] for a in finals_only}
    assert consumed <= all_ids                 # intermediates present in full list
    assert consumed.isdisjoint(final_ids)      # but hidden when filtered
    assert out_id in final_ids                 # deliverable remains
