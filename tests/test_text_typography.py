"""MED/030 Task #16C-B: title-card and lower-third typography, literal text and bounds."""

from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent import render, reuse
from app.agent.planner import PlanValidationError, validate_plan
from app.agent.render import compile_execution_plan
from app.agent.schemas import Plan
from app.capabilities import registry
from app.capabilities.media_options import TEXT_FONT_FILES
from app.capabilities.validation import CapabilityInputValidationError, validate_capability_input
from app.core.errors import ValidationError
from app.tools import imagemagick
from app.tools.imagemagick import DEFAULT_FONT, im_text

API = "/api/v1"
NEED_IM = pytest.mark.skipif(
    not (shutil.which("convert") and shutil.which("ffmpeg")),
    reason="ImageMagick and FFmpeg are required",
)
TITLE = registry.get("media.titlecard")
LOWER = registry.get("video.lowerthird")
LONG_GREEK_NAME = "Δρ. Αικατερίνη Παπαδοπούλου-Χατζηγεωργίου, Διευθύντρια"


def _ink(path: Path) -> tuple[int, int, int, int]:
    """Width, height, x, y of everything that is not the flat background."""
    out = subprocess.run(
        ["convert", str(path), "-trim", "-format", "%w %h %X %Y", "info:"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.split()
    return int(out[0]), int(out[1]), int(out[2]), int(out[3])


def _pixels(path: Path) -> bytes:
    return subprocess.run(
        ["convert", str(path), "-depth", "8", "rgb:-"],
        check=True, capture_output=True, timeout=30,
    ).stdout


async def _card(tmp_path: Path, name: str, text: str, **style) -> Path:
    out = tmp_path / f"{name}.png"
    style.setdefault("width", 320)
    style.setdefault("height", 180)
    await imagemagick.render_text_card(str(out), text, **style)
    return out


async def _lower(tmp_path: Path, name: str, text: str, caption: str | None = None,
                 **style) -> Path:
    out = tmp_path / f"{name}.png"
    await imagemagick.render_lowerthird(str(out), text, caption, **style)
    return out


# --- literal text (no binaries) ------------------------------------------------------


@pytest.mark.parametrize(("text", "argument"), [
    ("Hello", "Hello"),
    ("Καλημέρα «κόσμε»", "Καλημέρα «κόσμε»"),
    ("%w", "%%w"),
    ("%[fx:6*7]", "%%[fx:6*7]"),
    ("100%", "100%%"),
    ("@museum", "\\@museum"),
    ("a@b", "a@b"),
    ("C:\\new", "C:\\\\new"),
    ("@%\\", "\\@%%\\\\"),
    ("line one\nline two", "line one\nline two"),
])
def test_im_text_makes_user_text_literal(text: str, argument: str) -> None:
    assert im_text(text) == argument


async def test_text_is_escaped_exactly_once(monkeypatch) -> None:
    commands: list[list[str]] = []

    async def capture(args: list[str], **_kwargs) -> str:
        commands.append(args)
        return ""

    async def measure(args: list[str], **_kwargs) -> str:
        commands.append(args)
        return "100 40"

    monkeypatch.setattr(imagemagick.runner, "run", capture)
    monkeypatch.setattr(imagemagick.runner, "run_stdout", measure)
    await imagemagick.render_text_card("/o.png", "50%", width=320, height=180)
    await imagemagick.render_lowerthird("/o.png", "@me", "50%")
    assert commands[0][-2] == "caption:50%%"
    assert "\\@me" in commands[-1] and "50%%" in commands[-1]
    assert "label:\\@me" in commands[1] and "label:50%%" in commands[2]


# --- schema contract -------------------------------------------------------------------


def _valid(capability, **params) -> None:
    validate_capability_input(capability, params)


def _invalid(capability, **params) -> None:
    with pytest.raises(CapabilityInputValidationError):
        validate_capability_input(capability, params)


@pytest.mark.parametrize("font", sorted(TEXT_FONT_FILES))
def test_every_font_id_is_accepted(font: str) -> None:
    _valid(TITLE, text="T", font=font)
    _valid(LOWER, text="T", font=font)


@pytest.mark.parametrize("font", ["Comic Sans", "DejaVu-Sans", "dejavu-mono",
                                  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", ""])
def test_other_fonts_and_paths_are_rejected(font: str) -> None:
    _invalid(TITLE, text="T", font=font)
    _invalid(LOWER, text="T", font=font)


def test_size_and_alignment_ranges() -> None:
    for size in (12, 400):
        _valid(TITLE, text="T", font_size=size)
    for size in (11, 401, 30.5, True):
        _invalid(TITLE, text="T", font_size=size)
    for align in ("left", "center", "right"):
        _valid(TITLE, text="T", text_align=align)
    for align in ("top", "middle", "bottom"):
        _valid(TITLE, text="T", vertical_align=align)
    _invalid(TITLE, text="T", text_align="justify")
    _invalid(TITLE, text="T", vertical_align="center")
    for size in (12, 64):
        _valid(LOWER, text="T", text_size=size)
    for size in (11, 65):
        _invalid(LOWER, text="T", text_size=size)
    for size in (10, 48):
        _valid(LOWER, text="T", caption_size=size)
    for size in (9, 49):
        _invalid(LOWER, text="T", caption_size=size)


@pytest.mark.parametrize("colour", ["#fff", "#101418", "#ABCDEF", "white", "transparent",
                                    "DarkSlateGray"])
def test_simple_colours_are_accepted(colour: str) -> None:
    _valid(TITLE, text="T", background=colour, foreground=colour)
    _valid(LOWER, text="T", background=colour, foreground=colour)


@pytest.mark.parametrize("colour", ["rgb(1,2,3)", "red[50x20]", "#12345", "#1234567",
                                    "@colours.txt", "xc:red", "white ", "light blue",
                                    "red;blue", "", "srgb(0%,0%,0%)"])
def test_colour_expressions_are_rejected(colour: str) -> None:
    for field in ("background", "foreground"):
        _invalid(TITLE, text="T", **{field: colour})
        _invalid(LOWER, text="T", **{field: colour})


def test_text_lengths_and_title_card_size_limits() -> None:
    _valid(TITLE, text="x" * 500)
    _invalid(TITLE, text="x" * 501)
    _valid(LOWER, text="x" * 200, caption="y" * 300)
    _invalid(LOWER, text="x" * 201)
    _invalid(LOWER, text="x", caption="y" * 301)
    _valid(TITLE, text="T", width=3840, height=2160)
    _valid(TITLE, text="T", width=2160, height=3840)
    _valid(TITLE, text="T", width=3840)            # default height 1080
    _invalid(TITLE, text="T", width=3840, height=3840)  # area
    _invalid(TITLE, text="T", width=3841, height=100)
    _invalid(TITLE, text="T", width=100, height=3841)


async def test_renderers_reject_unsafe_values_before_running(monkeypatch) -> None:
    async def forbidden(*_args, **_kwargs):
        raise AssertionError("ImageMagick must not run")

    monkeypatch.setattr(imagemagick.runner, "run", forbidden)
    monkeypatch.setattr(imagemagick.runner, "run_stdout", forbidden)
    for bad in ({"background": "red[50x20]"}, {"foreground": "rgb(1,2,3)"},
                {"font": "/etc/passwd"}, {"text_size": 65}, {"caption_size": 9}):
        with pytest.raises(ValidationError):
            await imagemagick.render_lowerthird("/o.png", "Name", "Role", **bad)
    for bad in ({"background": "xc:red"}, {"font": "Comic Sans"}, {"font_size": 401},
                {"text_align": "justify"}, {"width": 3840, "height": 3840}):
        style = {"width": 320, "height": 180, **bad}
        with pytest.raises(ValidationError):
            await imagemagick.render_text_card("/o.png", "Title", **style)


async def test_unstyled_commands_are_unchanged(monkeypatch) -> None:
    commands: list[list[str]] = []

    async def capture(args: list[str], **_kwargs) -> str:
        commands.append(args)
        return ""

    async def measure(_args: list[str], **_kwargs) -> str:
        return "200 44"

    monkeypatch.setattr(imagemagick.runner, "run", capture)
    monkeypatch.setattr(imagemagick.runner, "run_stdout", measure)
    await imagemagick.render_text_card("/o.png", "Hello", width=320, height=180)
    await imagemagick.render_lowerthird("/l.png", "Name", "Role")
    assert commands[0] == [
        "convert", "-size", "320x180", "-background", "#101418", "-fill", "#ffffff",
        "-gravity", "center", "-font", DEFAULT_FONT, "caption:Hello", "/o.png",
    ]
    assert commands[1] == [
        "convert", "-size", "640x160", "xc:#101418", "-gravity", "northwest",
        "-fill", "#ffffff", "-font", DEFAULT_FONT, "-pointsize", "36",
        "-annotate", "+20+16", "Name", "-pointsize", "22", "-annotate", "+20+72", "Role",
        "/l.png",
    ]


# --- Agent ------------------------------------------------------------------------------

VID = "33333333-3333-3333-3333-333333333333"
_FACTS = {VID: {
    "media_type": "video", "extension": "mp4", "checksum_sha256": "e" * 64,
    "technical_metadata_status": "available", "width": 1280, "height": 720,
    "has_audio_stream": True,
}}


def _plan(params: dict) -> Plan:
    return Plan.model_validate({"type": "plan", "steps": [
        {"id": "lower", "capability_id": "video.lowerthird", "asset": VID, "params": params},
    ]})


@pytest.mark.parametrize("style", [{"font": "dejavu-serif-bold"}, {"text_size": 48},
                                   {"caption_size": 30}])
def test_styled_lower_thirds_are_not_fused_and_keep_their_fields(style: dict) -> None:
    plain = compile_execution_plan(_plan({"text": "Name"}), _FACTS)
    assert [node["kind"] for node in plain["nodes"]] == ["render_group"]
    styled_plan = _plan({"text": "Name", **style})
    styled = compile_execution_plan(styled_plan, _FACTS)
    assert all(node["kind"] != "render_group" for node in styled["nodes"])
    assert styled_plan.steps[0].params == {"text": "Name", **style}


def test_agent_plans_reject_unsafe_lower_third_values() -> None:
    scope = {VID: "video"}
    for bad in ({"background": "red[50x20]"}, {"font": "Comic Sans"}, {"text_size": 99}):
        with pytest.raises(PlanValidationError):
            validate_plan(_plan({"text": "Name", **bad}), scope)


def test_versions_change_only_for_the_lower_third_output() -> None:
    assert render.OPTIMIZER_VERSION == "agent-video-render-v2"
    assert reuse.REUSE_VERSIONS["video.lowerthird"] == "2"
    assert "media.titlecard" not in reuse.REUSE_VERSIONS
    assert reuse.REUSE_VERSIONS["video.subtitle.embed"] == "3"
    assert reuse.REUSE_VERSIONS["audio.mix"] == "2"


# --- real rendering ---------------------------------------------------------------------


@NEED_IM
async def test_default_renders_match_the_previous_commands(tmp_path: Path) -> None:
    old_card, old_lower = tmp_path / "old_card.png", tmp_path / "old_lower.png"
    subprocess.run(["convert", "-size", "320x180", "-background", "#101418", "-fill",
                    "#ffffff", "-gravity", "center", "-font", DEFAULT_FONT,
                    "caption:Ακρόπολη 2026", str(old_card)], check=True, timeout=30)
    subprocess.run(["convert", "-size", "640x160", "xc:#101418", "-gravity", "northwest",
                    "-fill", "#ffffff", "-font", DEFAULT_FONT, "-pointsize", "36",
                    "-annotate", "+20+16", "Μαρία Παπαδοπούλου", "-pointsize", "22",
                    "-annotate", "+20+72", "Επιμελήτρια", str(old_lower)],
                   check=True, timeout=30)
    card = await _card(tmp_path, "card", "Ακρόπολη 2026")
    lower = await _lower(tmp_path, "lower", "Μαρία Παπαδοπούλου", "Επιμελήτρια")
    assert _pixels(card) == _pixels(old_card)
    assert _pixels(lower) == _pixels(old_lower)


@NEED_IM
async def test_special_characters_print_literally(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("S", encoding="utf-8")
    widths = {}
    for name, text in {"fx": "%[fx:6*7]", "fortytwo": "42", "pct_w": "%w", "640": "640",
                       "at": f"@{secret}", "S": "S", "backslash": "C:\\new",
                       "real_newline": "C:\nnew", "one_line": "C:Xnew"}.items():
        if name in ("at", "S"):   # a long path: use a small caption under a short name
            widths[name] = _ink(await _lower(tmp_path, name, "N", text, caption_size=10))
        else:
            widths[name] = _ink(await _lower(tmp_path, name, text))
    assert widths["fx"][0] > 3 * widths["fortytwo"][0]          # not evaluated to 42
    assert widths["pct_w"][0] != widths["640"][0]               # not the canvas width
    assert widths["at"][0] > 5 * widths["S"][0]                 # the path, not the file
    # A typed backslash stays on one line; a real newline still breaks the line.
    assert widths["backslash"][1] < 1.5 * widths["one_line"][1]
    assert widths["real_newline"][1] > 1.5 * widths["one_line"][1]


@NEED_IM
async def test_fonts_render_differently_and_bold_is_heavier(tmp_path: Path) -> None:
    renders = {font: _pixels(await _card(tmp_path, font, "Ag Ωω ά", font=font,
                                         font_size=40))
               for font in TEXT_FONT_FILES}
    assert len(set(renders.values())) == 4

    def ink_amount(data: bytes) -> int:
        return sum(1 for value in data[::3] if value > 128)

    for family in ("dejavu-sans", "dejavu-serif"):
        assert ink_amount(renders[f"{family}-bold"]) > 1.15 * ink_amount(renders[family])


@NEED_IM
@pytest.mark.parametrize(("vertical", "horizontal"), [
    (v, h) for v in ("top", "middle", "bottom") for h in ("left", "center", "right")
])
async def test_title_alignment(tmp_path: Path, vertical: str, horizontal: str) -> None:
    width, height, x, y = _ink(await _card(
        tmp_path, f"{vertical}-{horizontal}", "Τίτλος", font_size=30,
        text_align=horizontal, vertical_align=vertical,
    ))
    centre_x, centre_y = x + width / 2, y + height / 2
    expected_x = {"left": x < 20, "center": abs(centre_x - 160) < 20,
                  "right": x + width > 300}
    expected_y = {"top": y < 20, "middle": abs(centre_y - 90) < 20,
                  "bottom": y + height > 160}
    assert expected_x[horizontal] and expected_y[vertical], (x, y, width, height)


@NEED_IM
async def test_explicit_title_sizes_and_overflow(tmp_path: Path) -> None:
    small = _ink(await _card(tmp_path, "s12", "Ab", font_size=12))
    large = _ink(await _card(tmp_path, "s400", "Ab", font_size=400, width=1000,
                             height=600))
    assert large[1] > 20 * small[1] / 2
    # Explicit sizes are never shrunk: too much text fails instead of being cut.
    with pytest.raises(ValidationError, match="does not fit"):
        await _card(tmp_path, "over", "Ένα δύο τρία τέσσερα πέντε έξι επτά", font_size=60)
    assert not (tmp_path / "over.png").exists()
    # caption: would cut a word wider than the card, so such a word is rejected.
    with pytest.raises(ValidationError, match="word in the title is too wide"):
        await _card(tmp_path, "word", "Supercalifragilistic", font_size=60, width=300,
                    height=400)
    with pytest.raises(ValidationError, match="too wide"):
        await _card(tmp_path, "narrow", "W", font_size=200, width=150, height=400)
    # Words that fit wrap onto lines, and nothing reaches past the edges.
    wrapped = _ink(await _card(tmp_path, "wrapped", "Super cali fragi listic", font_size=60,
                               width=300, height=400))
    assert wrapped[0] < 300 and wrapped[1] > 100 and wrapped[2] > 0
    # Without a size the text still fills the card, however long.
    auto = _ink(await _card(tmp_path, "auto", "Μουσείο " * 40))
    assert auto[0] <= 320 and auto[1] <= 180


@NEED_IM
async def test_lower_third_sizes_layout_and_overflow(tmp_path: Path) -> None:
    default = _ink(await _lower(tmp_path, "d", "Name", "Role"))
    bigger = _ink(await _lower(tmp_path, "b", "Name", "Role", text_size=64, caption_size=22))
    assert bigger[1] > default[1]
    # Two-line names fit without a caption; three caption lines fit below a name.
    await _lower(tmp_path, "two", "Μαρία\nΠαπαδοπούλου")
    await _lower(tmp_path, "notes", "Name", "Μουσείο\nΑθήνα\n2026")
    for kwargs, message in [
        ({"text": LONG_GREEK_NAME}, "name does not fit"),
        ({"text": "Name", "caption": "Curator " * 12}, "caption does not fit"),
        ({"text": "Name", "caption": "a\nb\nc\nd"}, "caption does not fit"),
        ({"text": "Two\nlines", "caption": "Role"}, "name does not fit"),
        ({"text": "Name", "caption": "Role", "text_size": 64, "caption_size": 48},
         "caption does not fit"),
    ]:
        with pytest.raises(ValidationError, match=message):
            await imagemagick.render_lowerthird(str(tmp_path / "x.png"), **kwargs)


# --- jobs -------------------------------------------------------------------------------


async def _job(client: AsyncClient, body: dict) -> tuple[int, dict]:
    response = await client.post(f"{API}/jobs", json=body)
    if response.status_code != 201:
        return response.status_code, response.json()
    return 201, (await client.get(f"{API}/jobs/{response.json()['id']}")).json()


async def _asset_ids(client: AsyncClient, pid: str) -> set[str]:
    return {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}


def _dims(data: bytes, tmp_path: Path) -> tuple[int, int]:
    path = tmp_path / "out.mp4"
    path.write_bytes(data)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.strip().split(",")
    return int(out[0]), int(out[1])


@pytest.mark.parametrize("params", [
    {"text": "T", "font": "Comic Sans"},
    {"text": "T", "background": "red[50x20]"},
    {"text": "T", "width": 3840, "height": 3840},
    {"text": "x" * 501},
    {"text": "T", "font_size": 500},
])
async def test_invalid_title_requests_get_422(client: AsyncClient, make_project,
                                               params: dict) -> None:
    pid = await make_project(f"title-422-{sorted(params)}")
    status, body = await _job(client, {"capability_id": "media.titlecard",
                                       "project_id": pid, "input": params})
    assert status == 422, body
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0


@NEED_IM
@pytest.mark.parametrize(("width", "height"), [(320, 180), (180, 320), (240, 240)])
async def test_styled_title_card_jobs_keep_their_shape(
    client: AsyncClient, make_project, tmp_path: Path, width: int, height: int
) -> None:
    pid = await make_project(f"title-{width}x{height}")
    status, job = await _job(client, {
        "capability_id": "media.titlecard", "project_id": pid,
        "input": {"text": "Έκθεση 100%", "duration": 0.5, "width": width, "height": height,
                  "font": "dejavu-serif-bold", "font_size": 24, "text_align": "left",
                  "vertical_align": "top", "background": "#000", "foreground": "white"},
    })
    assert status == 201 and job["status"] == "succeeded", job.get("error")
    data = (await client.get(job["output"]["outputs"][0]["download_path"])).content
    assert _dims(data, tmp_path) == (width, height)


@NEED_IM
async def test_overflowing_text_fails_the_job_without_an_asset(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("typography-overflow")
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=gray:s=640x360:d=1", "-pix_fmt", "yuv420p", str(clip)],
                   check=True, timeout=30)
    video = (await client.post(f"{API}/projects/{pid}/assets", files={
        "file": ("clip.mp4", io.BytesIO(clip.read_bytes()), "video/mp4")})).json()["id"]
    before = await _asset_ids(client, pid)
    for body in (
        {"capability_id": "video.lowerthird", "asset_id": video,
         "input": {"text": LONG_GREEK_NAME, "duration": 0.5}},
        {"capability_id": "media.titlecard",
         "input": {"text": "A very long title " * 5, "font_size": 100, "duration": 0.5,
                   "width": 320, "height": 180}},
    ):
        status, job = await _job(client, {"project_id": pid, **body})
        assert status == 201 and job["status"] == "failed" and job["output"] is None
        assert "does not fit" in job["error"]
    assert await _asset_ids(client, pid) == before


@NEED_IM
async def test_styled_lower_third_job_renders_on_the_video(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid = await make_project("typography-lower")
    clip = tmp_path / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                    "color=c=gray:s=1280x720:d=1", "-pix_fmt", "yuv420p", str(clip)],
                   check=True, timeout=30)
    video = (await client.post(f"{API}/projects/{pid}/assets", files={
        "file": ("clip.mp4", io.BytesIO(clip.read_bytes()), "video/mp4")})).json()["id"]
    status, job = await _job(client, {
        "capability_id": "video.lowerthird", "project_id": pid, "asset_id": video,
        "input": {"text": "Μαρία 50%", "caption": "@museum", "duration": 1,
                  "font": "dejavu-sans-bold", "text_size": 48, "caption_size": 28,
                  "background": "navy", "foreground": "#ff0"},
    })
    assert status == 201 and job["status"] == "succeeded", job.get("error")
    out = tmp_path / "out.mp4"
    out.write_bytes((await client.get(job["output"]["outputs"][0]["download_path"])).content)
    frame = tmp_path / "frame.png"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "0.5", "-i", str(out),
                    "-frames:v", "1", str(frame)], check=True, timeout=30)
    banner = subprocess.run(
        ["convert", str(frame), "-crop", "640x160+20+540", "-format",
         "%[fx:mean.r] %[fx:mean.b]", "info:"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout.split()
    assert float(banner[1]) > float(banner[0]) + 0.2   # navy banner over grey video
