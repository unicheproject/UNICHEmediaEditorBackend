"""MED/030: optional, validated styling for burned-in subtitles (video.subtitle.embed)."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest
from httpx import AsyncClient

from app.agent import reuse
from app.capabilities import registry
from app.capabilities.validation import CapabilityInputValidationError, validate_capability_input
from app.core.errors import ValidationError
from app.tools import ffmpeg
from tests.test_composition_clip_timing import NEED_FFMPEG

API = "/api/v1"
W, H = 320, 180
BACKGROUND = 0x20


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True,
                   timeout=30)


def _video(path: Path) -> Path:
    _ff("-f", "lavfi", "-i", f"color=c=0x202020:s={W}x{H}:r=10:d=2",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path))
    return path


def _srt(path: Path, text: str = "HELLO WORLD", start: str = "00:00:00,000",
         end: str = "00:00:02,000") -> Path:
    path.write_text(f"1\n{start} --> {end}\n{text}\n", encoding="utf-8")
    return path


def _frame(path: Path, seconds: float = 1.0) -> list[tuple[int, int, int]]:
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(seconds), "-i", str(path), "-frames:v", "1",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    return [(raw[i], raw[i + 1], raw[i + 2]) for i in range(0, len(raw), 3)]


def _text_pixels(frame: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    """(x, y) of pixels clearly different from the flat dark background."""
    return [(i % W, i // W) for i, (r, g, b) in enumerate(frame)
            if max(abs(r - BACKGROUND), abs(g - BACKGROUND), abs(b - BACKGROUND)) > 60]


def _count(frame: list[tuple[int, int, int]], test) -> int:
    return sum(1 for pixel in frame if test(*pixel))


async def _burn(tmp_path: Path, name: str, style: dict | None = None,
                srt: Path | None = None) -> Path:
    video = tmp_path / "video.mp4"
    if not video.exists():
        _video(video)
    subtitle = srt or (tmp_path / "subs.srt")
    if srt is None and not subtitle.exists():
        _srt(subtitle)
    out = tmp_path / f"{name}.mp4"
    await ffmpeg.subtitle_embed(str(video), str(subtitle), str(out), mode="burn", style=style)
    return out


def _libass_log(tmp_path: Path, style: dict | None, srt: Path) -> str:
    """libass's own report of the font it selected (and of any missing glyphs)."""
    video = tmp_path / "video.mp4"
    if not video.exists():
        _video(video)
    return subprocess.run(
        ["ffmpeg", "-v", "verbose", "-i", str(video), "-vf",
         ffmpeg.subtitle_burn_filter(str(srt), style), "-frames:v", "1", "-f", "null", "-"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stderr


# --- contract and validation (no binaries) ------------------------------------------


def test_schema_documents_the_optional_style_fields() -> None:
    schema = registry.get("video.subtitle.embed").input_schema
    props = schema["properties"]
    assert schema["required"] == ["subtitle_asset_id"]
    assert props["mode"]["default"] == "soft"
    assert props["font"]["enum"] == [
        "dejavu-sans", "dejavu-sans-bold", "dejavu-serif", "dejavu-serif-bold"
    ]
    assert (props["font_size"]["minimum"], props["font_size"]["maximum"]) == (12, 96)
    assert (props["outline"]["minimum"], props["outline"]["maximum"]) == (0, 4)
    assert (props["margin"]["minimum"], props["margin"]["maximum"]) == (0, 200)
    assert props["position"]["enum"] == ["bottom", "top"]
    for name in ("font", "font_size", "text_colour", "outline_colour", "outline",
                 "position", "margin"):
        assert "default" not in props[name]
        assert props[name]["x-help"]


@pytest.mark.parametrize("params", [
    {"font": "dejavu-sans"}, {"font": "dejavu-serif-bold", "font_size": 12},
    {"font_size": 96, "outline": 0, "margin": 200}, {"text_colour": "#aBcDeF"},
    {"outline_colour": "#000000", "outline": 4, "position": "top", "margin": 0},
])
def test_valid_burn_styles_are_accepted(params: dict) -> None:
    validate_capability_input(registry.get("video.subtitle.embed"),
                              {"subtitle_asset_id": "11111111-1111-1111-1111-111111111111",
                               "mode": "burn", **params})


@pytest.mark.parametrize("params", [
    {"font": "Arial"}, {"font": "/usr/share/fonts/x.ttf"}, {"font": "DejaVu Sans"},
    {"font_size": 11}, {"font_size": 97}, {"font_size": 24.5}, {"font_size": "24"},
    {"font_size": True}, {"text_colour": "red"}, {"text_colour": "#FFF"},
    {"text_colour": "#GG0000"}, {"outline_colour": "#0000001"}, {"outline": -1},
    {"outline": 5}, {"position": "middle"}, {"margin": -1}, {"margin": 201},
    {"text_colour": "#FF0000',force_style='Fontsize=99"},
])
def test_invalid_styles_are_rejected(params: dict) -> None:
    with pytest.raises(CapabilityInputValidationError):
        validate_capability_input(
            registry.get("video.subtitle.embed"),
            {"subtitle_asset_id": "11111111-1111-1111-1111-111111111111", "mode": "burn",
             **params},
            allow_step_refs=True,
        )


@pytest.mark.parametrize("mode", ["soft", None])
@pytest.mark.parametrize("field", ["font", "position", "margin"])
def test_style_without_burn_is_rejected(mode: str | None, field: str) -> None:
    value = {"font": "dejavu-sans", "position": "top", "margin": 10}[field]
    params: dict = {"subtitle_asset_id": "11111111-1111-1111-1111-111111111111", field: value}
    if mode is not None:
        params["mode"] = mode
    with pytest.raises(CapabilityInputValidationError, match="requires mode 'burn'"):
        validate_capability_input(registry.get("video.subtitle.embed"), params)


def test_force_style_mapping_and_colour_order() -> None:
    assert ffmpeg.subtitle_force_style({
        "font": "dejavu-sans-bold", "font_size": 30, "text_colour": "#FF8000",
        "outline_colour": "#123456", "outline": 2, "position": "top", "margin": 15,
    }) == ("FontName=DejaVu Sans,Bold=1,FontSize=30,PrimaryColour=&H000080FF,"
           "OutlineColour=&H00563412,Outline=2,Alignment=6,MarginV=15")
    with pytest.raises(ValidationError):
        ffmpeg.subtitle_force_style({"force_style": "Fontsize=99"})
    with pytest.raises(ValidationError):
        ffmpeg.subtitle_force_style({"text_colour": "#FF0000'"})


async def test_unstyled_commands_are_unchanged(monkeypatch) -> None:
    commands: list[list[str]] = []

    async def capture(args: list[str], **_kwargs) -> str:
        commands.append(args)
        return ""

    monkeypatch.setattr(ffmpeg.runner, "run", capture)
    await ffmpeg.subtitle_embed("/v.mp4", "/s/a.srt", "/o.mp4", mode="burn")
    await ffmpeg.subtitle_embed("/v.mp4", "/s/a.srt", "/o.mp4", mode="burn", style={})
    await ffmpeg.subtitle_embed("/v.mp4", "/s/a.srt", "/o.mp4", mode="soft")
    burn = [*ffmpeg._BASE, "-i", "/v.mp4", "-vf", "subtitles='/s/a.srt'",
            "-pix_fmt", "yuv420p", "/o.mp4"]
    assert commands[0] == burn and commands[1] == burn
    assert commands[2] == [*ffmpeg._BASE, "-i", "/v.mp4", "-i", "/s/a.srt", "-c", "copy",
                           "-c:s", "mov_text", "/o.mp4"]
    with pytest.raises(ValidationError, match="burn mode"):
        await ffmpeg.subtitle_embed("/v.mp4", "/s/a.srt", "/o.mp4", mode="soft",
                                    style={"font": "dejavu-sans"})


def test_reuse_version_is_bumped() -> None:
    assert reuse.REUSE_VERSIONS["video.subtitle.embed"] == "2"


# --- visible rendering (tiny fixtures) -----------------------------------------------


@NEED_FFMPEG
async def test_default_burn_is_white_text_at_the_bottom(tmp_path: Path) -> None:
    frame = _frame(await _burn(tmp_path, "default"))
    pixels = _text_pixels(frame)
    assert len(pixels) > 100
    assert sum(y for _x, y in pixels) / len(pixels) > H * 0.7
    # libass default: small white letters (yuv420p blurs thin strokes a little).
    assert _count(frame, lambda r, g, b: min(r, g, b) > 180) > 20


@pytest.mark.parametrize(("font", "font_file"), [
    ("dejavu-sans", "DejaVuSans.ttf"), ("dejavu-sans-bold", "DejaVuSans-Bold.ttf"),
    ("dejavu-serif", "DejaVuSerif.ttf"), ("dejavu-serif-bold", "DejaVuSerif-Bold.ttf"),
])
@NEED_FFMPEG
def test_selected_font_is_the_one_libass_uses(tmp_path: Path, font: str, font_file: str) -> None:
    log = _libass_log(tmp_path, {"font": font}, _srt(tmp_path / "subs.srt"))
    selected = [line for line in log.splitlines() if "fontselect" in line]
    assert selected and all(f"/{font_file}" in line for line in selected), selected


@NEED_FFMPEG
async def test_fonts_render_differently(tmp_path: Path) -> None:
    sans = _text_pixels(_frame(await _burn(tmp_path, "sans", {"font": "dejavu-sans"})))
    serif = _text_pixels(_frame(await _burn(tmp_path, "serif", {"font": "dejavu-serif-bold"})))
    assert set(sans) != set(serif)


@NEED_FFMPEG
async def test_font_size_changes_the_text_height(tmp_path: Path) -> None:
    def height(pixels: list[tuple[int, int]]) -> int:
        ys = [y for _x, y in pixels]
        return max(ys) - min(ys)

    small = _text_pixels(_frame(await _burn(tmp_path, "small", {"font_size": 12})))
    large = _text_pixels(_frame(await _burn(tmp_path, "large", {"font_size": 48})))
    assert height(large) > 2.5 * height(small)


@NEED_FFMPEG
async def test_text_and_outline_colours(tmp_path: Path) -> None:
    # Large letters, so chroma subsampling (yuv420p) keeps their colour clear.
    is_red = lambda r, g, b: r - max(g, b) > 80  # noqa: E731
    is_green = lambda r, g, b: g - max(r, b) > 80  # noqa: E731
    red = _frame(await _burn(tmp_path, "red",
                             {"text_colour": "#FF0000", "outline": 0, "font_size": 40}))
    assert _count(red, is_red) > 100
    assert _count(red, lambda r, g, b: min(r, g, b) > 180) < 5               # not white

    outline: dict[int, list[tuple[int, int, int]]] = {}
    for width in (4, 1, 0):
        outline[width] = _frame(await _burn(tmp_path, f"outline{width}", {
            "outline_colour": "#00FF00", "outline": width, "font_size": 40,
        }))
    assert _count(outline[4], is_green) > 100
    assert _count(outline[4], is_green) > _count(outline[1], is_green)
    assert _count(outline[0], is_green) < 5                                 # no outline


@NEED_FFMPEG
async def test_position_and_margin(tmp_path: Path) -> None:
    top = _text_pixels(_frame(await _burn(tmp_path, "top", {"position": "top"})))
    bottom = _text_pixels(_frame(await _burn(tmp_path, "bottom", {"position": "bottom"})))
    assert max(y for _x, y in top) < H / 3 and min(y for _x, y in bottom) > 2 * H / 3
    for pixels in (top, bottom):                                        # both centred
        assert abs(sum(x for x, _y in pixels) / len(pixels) - W / 2) < W * 0.08
    near = _text_pixels(_frame(await _burn(tmp_path, "near", {"margin": 0})))
    far = _text_pixels(_frame(await _burn(tmp_path, "far", {"margin": 100})))
    assert max(y for _x, y in far) < max(y for _x, y in near) - 40          # lifted up
    top_far = _text_pixels(_frame(await _burn(tmp_path, "topfar",
                                              {"position": "top", "margin": 100})))
    assert min(y for _x, y in top_far) > min(y for _x, y in top) + 40      # pushed down


@NEED_FFMPEG
async def test_style_only_inside_the_cue_interval(tmp_path: Path) -> None:
    srt = _srt(tmp_path / "timed.srt", start="00:00:00,500", end="00:00:01,500")
    out = await _burn(tmp_path, "timed", {"text_colour": "#FF0000", "outline": 0,
                                          "font_size": 40}, srt=srt)
    assert len(_text_pixels(_frame(out, 0.2))) == 0
    assert len(_text_pixels(_frame(out, 1.0))) > 50
    assert len(_text_pixels(_frame(out, 1.8))) == 0


@NEED_FFMPEG
async def test_greek_text_renders_without_missing_glyphs(tmp_path: Path) -> None:
    text = "Καλημέρα, κόσμε: «δοκιμή» ΆΈΉΊΌΎΏ ϊΰ"
    srt = _srt(tmp_path / "greek.srt", text=text)
    style = {"font": "dejavu-serif", "font_size": 20}
    log = _libass_log(tmp_path, style, srt)
    assert "Glyph" not in log and "not found" not in log.lower()
    out = await _burn(tmp_path, "greek", style, srt=srt)
    assert len(_text_pixels(_frame(out))) > 200


@NEED_FFMPEG
async def test_long_lines_with_punctuation_do_not_break_the_filter(tmp_path: Path) -> None:
    text = ("Ένα πολύ μεγάλο κείμενο, με κόμματα, άνω-κάτω τελείες: ερωτηματικά; "
            "εισαγωγικά 'μονά' και \"διπλά\", [αγκύλες] και \\ ανάποδες καθέτους") * 2
    srt = _srt(tmp_path / "long.srt", text=text)
    out = await _burn(tmp_path, "long", {"font_size": 14, "outline": 1, "margin": 5}, srt=srt)
    assert len(_text_pixels(_frame(out))) > 300


# --- the public capability ----------------------------------------------------------


async def _assets(client: AsyncClient, make_project, tmp_path: Path, name: str,
                  subtitle: bytes) -> tuple[str, str, str]:
    pid = await make_project(name)
    video = (await client.post(f"{API}/projects/{pid}/assets", files={
        "file": ("clip.mp4", io.BytesIO(_video(tmp_path / "v.mp4").read_bytes()), "video/mp4")
    })).json()["id"]
    subs = (await client.post(f"{API}/projects/{pid}/assets", files={
        "file": ("subs.srt", io.BytesIO(subtitle), "application/x-subrip")
    })).json()["id"]
    return pid, video, subs


async def _submit(client: AsyncClient, pid: str, video: str, subs: str, extra: dict):
    return await client.post(f"{API}/jobs", json={
        "capability_id": "video.subtitle.embed", "project_id": pid, "asset_id": video,
        "input": {"subtitle_asset_id": subs, **extra},
    })


@NEED_FFMPEG
async def test_styled_burn_job_publishes_a_styled_video(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid, video, subs = await _assets(client, make_project, tmp_path, "subs-styled",
                                     "1\n00:00:00,000 --> 00:00:02,000\nΓΕΙΑ\n".encode())
    response = await _submit(client, pid, video, subs, {
        "mode": "burn", "font": "dejavu-sans-bold", "text_colour": "#FF0000", "outline": 0,
        "position": "top", "font_size": 40,
    })
    assert response.status_code == 201, response.text
    job = (await client.get(f"{API}/jobs/{response.json()['id']}")).json()
    assert job["status"] == "succeeded", job.get("error")
    out = tmp_path / "out.mp4"
    out.write_bytes((await client.get(job["output"]["outputs"][0]["download_path"])).content)
    frame = _frame(out)
    assert _count(frame, lambda r, g, b: r - max(g, b) > 80) > 100
    assert max(y for _x, y in _text_pixels(frame)) < H / 2


@pytest.mark.parametrize("extra", [
    {"mode": "soft", "font": "dejavu-sans"},
    {"font_size": 30},                       # mode defaults to soft
    {"mode": "burn", "font": "Comic Sans"},
    {"mode": "burn", "text_colour": "white"},
    {"mode": "burn", "margin": 500},
])
async def test_invalid_requests_return_422_before_any_job(
    client: AsyncClient, make_project, tmp_path: Path, extra: dict
) -> None:
    pid = await make_project(f"subs-422-{sorted(extra)}")
    video = (await client.post(f"{API}/projects/{pid}/assets", files={
        "file": ("clip.mp4", io.BytesIO(b"x"), "video/mp4")})).json()["id"]
    subs = (await client.post(f"{API}/projects/{pid}/assets", files={
        "file": ("subs.srt", io.BytesIO(b"x"), "application/x-subrip")})).json()["id"]
    response = await _submit(client, pid, video, subs, extra)
    assert response.status_code == 422, response.text
    assert (await client.get(f"{API}/projects/{pid}/jobs")).json()["total"] == 0


@NEED_FFMPEG
async def test_unreadable_subtitles_fail_without_an_asset(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid, video, subs = await _assets(client, make_project, tmp_path, "subs-broken",
                                     b"this is not a subtitle\n")
    before = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    response = await _submit(client, pid, video, subs, {"mode": "burn", "font": "dejavu-sans"})
    assert response.status_code == 201, response.text
    job = (await client.get(f"{API}/jobs/{response.json()['id']}")).json()
    assert job["status"] == "failed" and job["output"] is None
    after = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    assert after == before
