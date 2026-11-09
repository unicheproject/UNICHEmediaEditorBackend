"""MED/027: an optional audio track of positioned audio clips in compositions."""

from __future__ import annotations

import array
import hashlib
import io
import json
import math
import subprocess
import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient
from pydantic import ValidationError as PydanticValidationError

from app.api.v1.jobs import get_enqueuer
from app.capabilities.compose_contract import AUDIO_TRACK_MAX_CLIPS
from app.composition.document import canonical_document
from app.composition.render import AudioClipSpec, ComposeRenderRequest
from app.core.database import async_session_factory
from app.core.errors import ValidationError
from app.main import app
from app.models.asset import Asset
from app.models.job import Job
from app.schemas.composition import CompositionDocument
from app.services.jobs import execute_job
from app.tools import compose_renderer, ffmpeg
from app.tools.compose_policy import (
    DEFAULT_COMPOSE_RENDER_POLICY,
    LOSSLESS_COMPOSE_RENDER_POLICY,
)
from app.tools.ffmpeg import PositionedAudio
from tests.test_compose_soundtrack import _amplitude, _samples
from tests.test_composition_clip_timing import NEED_FFMPEG, _put
from tests.test_composition_render import _compiled

API = "/api/v1"
A = "11111111-1111-1111-1111-111111111111"
B = "22222222-2222-2222-2222-222222222222"
C = "33333333-3333-3333-3333-333333333333"


def _uid() -> str:
    return str(uuid.uuid4())


def _audio_clip(asset_id: str, timeline_start: object = 0.5, **fields: object) -> dict:
    return {"id": _uid(), "asset_id": asset_id, "timeline_start": timeline_start, **fields}


def _document(visual: list[dict], audio: list[dict] | None = None, *, bed: dict | None = None,
              audio_track: dict | None = None) -> dict:
    tracks: list[dict] = [{"id": _uid(), "kind": "video", "clips": visual}]
    if audio is not None:
        tracks.append({"id": _uid(), "kind": "audio", "clips": audio})
    if audio_track is not None:
        tracks.append(audio_track)
    return {"schema_version": 1, "canvas": {"width": 96, "height": 64},
            "tracks": tracks, "audio_bed": bed}


def _validate(raw: dict) -> CompositionDocument:
    return CompositionDocument.model_validate(raw)


# --- schema (no binaries) ---------------------------------------------------


@pytest.mark.parametrize("visual", [
    [{"id": _uid(), "asset_id": A}],
    [{"id": _uid(), "asset_id": A, "source_start": 1.0},
     {"id": _uid(), "kind": "image", "asset_id": B, "duration": 2.0}],
])
def test_documents_without_audio_track_keep_json_and_hash(visual: list[dict]) -> None:
    raw = _document(visual)
    data, digest = canonical_document(_validate(raw))
    assert data == raw
    legacy = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert digest == hashlib.sha256(legacy.encode("utf-8")).hexdigest()


def test_audio_track_serializes_canonically() -> None:
    clip = _audio_clip(C, 7, source_start=0.2, source_end=1.4, volume=0.6)
    plain = _audio_clip(C, 1)
    raw = _document([{"id": _uid(), "asset_id": A}], [clip, plain])
    data, _ = canonical_document(_validate(raw))
    track = data["tracks"][1]
    assert track["kind"] == "audio"
    assert track["clips"][0] == {**clip, "timeline_start": 7.0}
    # Volume defaults to 1.0 and is always stored, so omitting it is the same document.
    assert track["clips"][1] == {**plain, "timeline_start": 1.0, "volume": 1.0}
    explicit = json.loads(json.dumps(raw))
    explicit["tracks"][1]["clips"][1]["volume"] = 1.0
    assert canonical_document(_validate(explicit))[1] == canonical_document(_validate(raw))[1]


def test_audio_track_allows_up_to_64_clips() -> None:
    assert AUDIO_TRACK_MAX_CLIPS == 64
    visual = [{"id": _uid(), "asset_id": A}]
    _validate(_document(visual, [_audio_clip(C, i * 0.1) for i in range(64)]))
    with pytest.raises(PydanticValidationError):
        _validate(_document(visual, [_audio_clip(C, i * 0.1) for i in range(65)]))


def _bad_track_documents() -> list[dict]:
    video = {"id": _uid(), "kind": "video", "clips": [{"id": _uid(), "asset_id": A}]}
    audio = {"id": _uid(), "kind": "audio", "clips": [_audio_clip(C)]}
    base = {"schema_version": 1, "canvas": {"width": 96, "height": 64}, "audio_bed": None}
    shared_track = {**audio, "id": video["id"]}
    shared_clip = {**audio, "clips": [{**_audio_clip(C), "id": video["clips"][0]["id"]}]}
    return [
        {**base, "tracks": [audio]},                       # no video track
        {**base, "tracks": [audio, video]},                # audio before video
        {**base, "tracks": [video, video]},                # two video tracks
        {**base, "tracks": [video, audio, audio]},         # two audio tracks
        {**base, "tracks": [video, {**audio, "clips": []}]},
        {**base, "tracks": [video, {**audio, "kind": "music"}]},
        {**base, "tracks": [video, shared_track]},         # duplicate track id
        {**base, "tracks": [video, shared_clip]},          # duplicate clip id
    ]


@pytest.mark.parametrize("raw", _bad_track_documents())
def test_invalid_track_order_kind_or_count_is_rejected(raw: dict) -> None:
    with pytest.raises(PydanticValidationError):
        _validate(raw)


@pytest.mark.parametrize("fields", [
    {"timeline_start": -1}, {"timeline_start": math.inf}, {"timeline_start": math.nan},
    {"timeline_start": "1"}, {"timeline_start": True}, {"timeline_start": None},
    {"source_start": -0.1}, {"source_end": 0}, {"source_start": 2, "source_end": 2},
    {"source_start": 3, "source_end": 1}, {"source_end": math.inf},
    {"volume": -0.01}, {"volume": 1.01}, {"volume": math.nan}, {"volume": None},
    {"kind": "audio"}, {"duration": 2.0},
])
def test_invalid_audio_clip_fields_are_rejected(fields: dict) -> None:
    clip = {**_audio_clip(C), **fields}
    with pytest.raises(PydanticValidationError):
        _validate(_document([{"id": _uid(), "asset_id": A}], [clip]))


def test_missing_timeline_start_is_rejected() -> None:
    clip = {"id": _uid(), "asset_id": C}
    with pytest.raises(PydanticValidationError):
        _validate(_document([{"id": _uid(), "asset_id": A}], [clip]))


def test_openapi_describes_the_audio_track() -> None:
    schemas = app.openapi()["components"]["schemas"]
    clip = schemas["CompositionAudioClip"]
    assert set(clip["required"]) == {"id", "asset_id", "timeline_start"}
    assert clip["properties"]["volume"]["maximum"] == 1.0
    assert clip["properties"]["volume"]["default"] == 1.0
    track = schemas["CompositionAudioTrack"]
    assert track["properties"]["clips"]["maxItems"] == 64
    tracks = schemas["CompositionDocument"]["properties"]["tracks"]
    assert tracks["maxItems"] == 2
    refs = {item["$ref"].rsplit("/", 1)[1] for item in tracks["items"]["oneOf"]}
    assert refs == {"CompositionVideoTrack", "CompositionAudioTrack"}


async def test_renderer_passes_audio_clips_only_when_set(monkeypatch, tmp_path) -> None:
    calls: list[dict] = []

    async def fake_compose(_segments, _dst, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(compose_renderer.ffmpeg, "compose", fake_compose)
    ids = (uuid.UUID(A),)
    await compose_renderer.render_compose(
        ComposeRenderRequest(ids, 96, 64),
        compose_renderer.ComposeRenderSources(video_paths=("/v",)), "/o.mp4", work_dir=tmp_path,
    )
    spec = AudioClipSpec(uuid.UUID(C), 1.5, 0.2, 1.0, 0.5)
    await compose_renderer.render_compose(
        ComposeRenderRequest(ids, 96, 64, audio_clips=(spec,)),
        compose_renderer.ComposeRenderSources(video_paths=("/v",), audio_clip_paths=("/c",)),
        "/o.mp4", work_dir=tmp_path,
    )
    assert "audio_clips" not in calls[0]
    assert calls[1]["audio_clips"] == [PositionedAudio("/c", 1.5, 0.2, 1.0, 0.5)]
    with pytest.raises(ValidationError, match="audio clips"):
        await compose_renderer.render_compose(
            ComposeRenderRequest(ids, 96, 64, audio_clips=(spec,)),
            compose_renderer.ComposeRenderSources(video_paths=("/v",)),
            "/o.mp4", work_dir=tmp_path,
        )


async def test_compose_without_audio_track_uses_the_legacy_bed_path(
    monkeypatch, tmp_path
) -> None:
    calls: list[str] = []

    async def legacy_mix(*_args, **_kwargs) -> None:
        calls.append("audio_mix")

    async def new_mix(*_args, **_kwargs) -> None:
        calls.append("timeline")

    async def no_op(*_args, **_kwargs) -> str:
        return ""

    async def segment(_src, dst, *_args, **_kwargs) -> None:
        Path(dst).write_bytes(b"x")

    monkeypatch.setattr(ffmpeg, "_normalize_segment", segment)
    monkeypatch.setattr(ffmpeg.runner, "run", no_op)
    monkeypatch.setattr(ffmpeg, "audio_mix", legacy_mix)
    monkeypatch.setattr(ffmpeg, "_mix_timeline_audio", new_mix)
    out = str(tmp_path / "o.mp4")
    await ffmpeg.compose(["/v"], out, width=96, height=64, work_dir=tmp_path,
                         audio_path="/bed", audio_mode="replace")
    await ffmpeg.compose(["/v"], out, width=96, height=64, work_dir=tmp_path,
                         audio_path="/bed", audio_clips=[PositionedAudio("/c", 0.5)])
    assert calls == ["audio_mix", "timeline"]


# --- save, compile and resolve -------------------------------------------------


async def _upload(client: AsyncClient, pid: str, name: str, data: bytes = b"x") -> str:
    response = await client.post(
        f"{API}/projects/{pid}/assets",
        files={"file": (name, io.BytesIO(data), "application/octet-stream")},
    )
    assert response.status_code == 201, response.text
    return str(response.json()["id"])


async def _set_metadata(asset_id: str, **values: object) -> None:
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(asset_id))
        assert asset is not None
        asset.technical_metadata_status = "available"
        for key, value in values.items():
            setattr(asset, key, value)
        await session.commit()


async def test_audio_track_persists_and_compiles_in_order(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("audio-clips-compile")
    video = await _upload(client, pid, "clip.mp4")
    first = await _upload(client, pid, "voice.wav")
    second = await _upload(client, pid, "sfx.mp3")
    clips = [_audio_clip(second, 3, volume=0.4), _audio_clip(first, 0.5, source_end=2)]
    saved = await _put(client, pid, _document([{"id": _uid(), "asset_id": video}], clips))
    assert saved.status_code == 201, saved.text
    got = (await client.get(
        f"{API}/projects/{pid}/composition/revisions/{saved.json()['revision_id']}"
    )).json()["document"]["tracks"][1]["clips"]
    assert [(c["asset_id"], c["timeline_start"], c["volume"]) for c in got] == [
        (second, 3.0, 0.4), (first, 0.5, 1.0)
    ]
    request = (await _compiled(saved.json()["revision_id"])).request
    assert request.audio_clips == (
        AudioClipSpec(uuid.UUID(second), 3.0, 0.0, None, 0.4),
        AudioClipSpec(uuid.UUID(first), 0.5, 0.0, 2.0, 1.0),
    )
    assert [str(i) for i in request.video_asset_ids] == [video]


async def test_composition_without_audio_track_compiles_without_audio_clips(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("audio-clips-none")
    video = await _upload(client, pid, "clip.mp4")
    saved = (await _put(client, pid, _document([{"id": _uid(), "asset_id": video}]))).json()
    assert (await _compiled(saved["revision_id"])).request.audio_clips is None


@pytest.mark.parametrize("wrong", ["video", "image"])
async def test_audio_clips_must_reference_audio_assets(
    client: AsyncClient, make_project, wrong: str
) -> None:
    pid = await make_project(f"audio-clips-type-{wrong}")
    video = await _upload(client, pid, "clip.mp4")
    other = video if wrong == "video" else await _upload(client, pid, "photo.png")
    response = await _put(client, pid, _document(
        [{"id": _uid(), "asset_id": video}], [_audio_clip(other)]
    ))
    assert response.status_code == 422, response.text


async def test_the_bed_may_also_be_an_audio_clip(client: AsyncClient, make_project) -> None:
    pid = await make_project("audio-clips-bed-reuse")
    video = await _upload(client, pid, "clip.mp4")
    music = await _upload(client, pid, "music.wav")
    response = await _put(client, pid, _document(
        [{"id": _uid(), "asset_id": video}], [_audio_clip(music, 1)],
        bed={"asset_id": music, "mode": "mix", "volume": 0.3},
    ))
    assert response.status_code == 201, response.text


@pytest.mark.parametrize(("clip", "message"), [
    ({"timeline_start": 4.0}, "must start before the end of the video track"),
    ({"timeline_start": 1.0, "source_start": 5.0}, "start must be before the end of the audio"),
])
async def test_save_rejects_out_of_range_audio_when_metadata_is_known(
    client: AsyncClient, make_project, clip: dict, message: str
) -> None:
    pid = await make_project(f"audio-clips-bounds-{clip}")
    video = await _upload(client, pid, "clip.mp4")
    image = await _upload(client, pid, "photo.png")
    voice = await _upload(client, pid, "voice.wav")
    await _set_metadata(video, duration_seconds=3.0, frame_rate=25.0)
    await _set_metadata(voice, duration_seconds=2.0)
    visual = [{"id": _uid(), "asset_id": video, "source_start": 1.0},  # 2 s
              {"id": _uid(), "kind": "image", "asset_id": image, "duration": 2.0}]
    response = await _put(client, pid, _document(visual, [{**_audio_clip(voice), **clip}]))
    assert response.status_code == 422, response.text
    assert message in response.json()["error"]["message"]


async def test_save_defers_the_timeline_bound_when_lengths_are_unknown(
    client: AsyncClient, make_project
) -> None:
    pid = await make_project("audio-clips-deferred")
    video = await _upload(client, pid, "clip.mp4")  # metadata pending
    voice = await _upload(client, pid, "voice.wav")
    response = await _put(client, pid, _document(
        [{"id": _uid(), "asset_id": video}], [_audio_clip(voice, 500)]
    ))
    assert response.status_code == 201, response.text


# --- real rendering (tiny fixtures) -------------------------------------------


def _ff(*args: str) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", *args], check=True, capture_output=True,
                   timeout=30)


def _video(path: Path, seconds: float, frequency: int | None = 440) -> Path:
    args = ["-f", "lavfi", "-i", f"color=c=red:s=96x64:r=30:d={seconds}"]
    if frequency is not None:
        args += ["-f", "lavfi", "-i", f"sine=frequency={frequency}:duration={seconds}"]
    _ff(*args, "-c:v", "libx264", "-pix_fmt", "yuv420p", *(["-c:a", "aac"] if frequency else []),
        "-shortest", str(path))
    return path


def _tone(path: Path, frequency: int, seconds: float) -> Path:
    _ff("-f", "lavfi", "-i", f"sine=frequency={frequency}:sample_rate=44100:duration={seconds}",
        str(path))
    return path


def _durations(path: Path) -> dict[str, float]:
    data = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,codec_name:format=duration",
         "-count_packets", "-show_entries", "stream=duration", "-of", "json", str(path)],
        check=True, capture_output=True, timeout=30,
    ).stdout)
    result = {"format": float(data["format"]["duration"])}
    for stream in data["streams"]:
        if "duration" in stream:
            result[stream["codec_type"]] = float(stream["duration"])
    return result


def _codecs(path: Path) -> set[str]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name",
                          "-of", "csv=p=0", str(path)], check=True, capture_output=True,
                         text=True, timeout=30).stdout
    return set(out.split())


def _level(path: Path, start: float, length: float, frequency: int) -> float:
    return _amplitude(_samples(path, start, length), frequency)


def _onset(path: Path, near: float, threshold: float = 0.02) -> float:
    """First time (s) the decoded signal exceeds `threshold` after `near` - 0.2 s."""
    start = near - 0.2
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-ss", str(start), "-t", "0.4", "-map",
         "0:a:0", "-ac", "1", "-ar", "44100", "-f", "f32le", "-"],
        check=True, capture_output=True, timeout=30,
    ).stdout
    samples = array.array("f", raw)
    index = next(i for i, value in enumerate(samples) if abs(value) > threshold)
    return start + index / 44100


async def _compose(tmp_path: Path, segments: list[Path], clips: list[PositionedAudio], *,
                   bed: Path | None = None, mode: str = "mix", volume: float = 0.6,
                   policy=DEFAULT_COMPOSE_RENDER_POLICY, name: str = "out") -> Path:
    out = tmp_path / f"{name}{policy.suffix}"
    work = tmp_path / f"work-{name}"
    work.mkdir()
    await ffmpeg.compose([str(s) for s in segments], str(out), width=96, height=64,
                         work_dir=work, audio_path=str(bed) if bed else None, audio_mode=mode,
                         music_volume=volume, policy=policy, audio_clips=clips)
    return out


@NEED_FFMPEG
async def test_clip_starts_on_time_with_silence_around_it(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.mp4", 3, frequency=None)  # silent picture
    voice = _tone(tmp_path / "voice.wav", 1000, 0.8)
    out = await _compose(tmp_path, [video], [PositionedAudio(str(voice), 1.0)],
                         policy=LOSSLESS_COMPOSE_RENDER_POLICY)
    assert _onset(out, 1.0) == pytest.approx(1.0, abs=0.002)
    assert _level(out, 0.3, 0.6, 1000) < 0.001         # before
    assert _level(out, 1.2, 0.4, 1000) > 0.05          # during
    assert _level(out, 2.1, 0.6, 1000) < 0.001         # after (ends at 1.8 s)
    durations = _durations(out)
    assert durations["format"] == pytest.approx(3.0, abs=0.05)


@NEED_FFMPEG
async def test_source_window_and_volume_are_applied(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.mp4", 3, frequency=None)
    # 0-1 s at 1000 Hz then 1-2 s at 1500 Hz; keep only the 1500 Hz second.
    first, second = _tone(tmp_path / "a.wav", 1000, 1), _tone(tmp_path / "b.wav", 1500, 1)
    joined = tmp_path / "joined.wav"
    _ff("-i", str(first), "-i", str(second), "-filter_complex", "[0][1]concat=n=2:v=0:a=1",
        str(joined))
    out = await _compose(tmp_path, [video], [
        PositionedAudio(str(joined), 0.5, source_start=1.0, source_end=1.5),
        PositionedAudio(str(second), 2.0, volume=0.5),
    ], policy=LOSSLESS_COMPOSE_RENDER_POLICY)
    assert _level(out, 0.6, 0.3, 1500) > 0.05 and _level(out, 0.6, 0.3, 1000) < 0.002
    assert _level(out, 1.1, 0.3, 1500) < 0.002          # window ended at 1.0 s
    full, half = _level(out, 0.6, 0.3, 1500), _level(out, 2.2, 0.3, 1500)
    assert half / full == pytest.approx(0.5, abs=0.03)


@NEED_FFMPEG
async def test_overlapping_clips_add_without_changing_the_soundtrack(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.mp4", 3, frequency=440)
    low, high = _tone(tmp_path / "low.wav", 1200, 2), _tone(tmp_path / "high.wav", 1700, 2)
    out = await _compose(tmp_path, [video], [
        PositionedAudio(str(low), 0.5), PositionedAudio(str(high), 1.0, volume=0.5),
    ], policy=LOSSLESS_COMPOSE_RENDER_POLICY)
    soundtrack_alone = _level(out, 0.1, 0.3, 440)
    assert _level(out, 1.3, 0.4, 440) == pytest.approx(soundtrack_alone, rel=0.05)
    alone = _level(out, 0.6, 0.3, 1200)
    assert _level(out, 1.3, 0.4, 1200) == pytest.approx(alone, rel=0.05)   # not normalised
    assert _level(out, 1.3, 0.4, 1700) == pytest.approx(alone * 0.5, rel=0.08)


@NEED_FFMPEG
async def test_clip_past_the_end_is_trimmed_to_the_video(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.mp4", 2)
    voice = _tone(tmp_path / "voice.wav", 1000, 3)
    out = await _compose(tmp_path, [video], [PositionedAudio(str(voice), 1.5)])
    durations = _durations(out)
    assert durations["video"] == pytest.approx(2.0, abs=0.05)
    assert durations["audio"] == pytest.approx(2.0, abs=0.05)
    assert _level(out, 1.6, 0.3, 1000) > 0.05


@pytest.mark.parametrize("start", [2.0, 2.5])
@NEED_FFMPEG
async def test_clip_starting_at_or_after_the_end_fails(tmp_path: Path, start: float) -> None:
    video = _video(tmp_path / "v.mp4", 2)
    voice = _tone(tmp_path / "voice.wav", 1000, 1)
    with pytest.raises(ValidationError, match="must start before the end of the video track"):
        await _compose(tmp_path, [video], [PositionedAudio(str(voice), start)])


@pytest.mark.parametrize("mode", ["mix", "duck", "replace"])
@NEED_FFMPEG
async def test_bed_modes_keep_their_rules_and_clips_stay_audible(
    tmp_path: Path, mode: str
) -> None:
    video = _video(tmp_path / "v.mp4", 3, frequency=440)
    bed = _tone(tmp_path / "bed.wav", 1200, 3)
    voice = _tone(tmp_path / "voice.wav", 1700, 1)
    out = await _compose(tmp_path, [video], [PositionedAudio(str(voice), 1.0)],
                         bed=bed, mode=mode, volume=0.6)
    bed_alone = _level(out, 0.2, 0.6, 1200)
    if mode == "replace":
        assert _level(out, 0.2, 0.6, 440) < 0.002          # soundtrack replaced
    else:
        assert _level(out, 0.2, 0.6, 440) > 0.02
    assert bed_alone > 0.01
    assert _level(out, 1.2, 0.6, 1700) > 0.05              # clip on top in every mode
    assert _level(out, 1.2, 0.6, 1200) == pytest.approx(bed_alone, rel=0.06)
    # Same bed levels as a composition without an audio track.
    legacy = tmp_path / "legacy.mp4"
    legacy_work = tmp_path / "legacy-work"
    legacy_work.mkdir()
    await ffmpeg.compose([str(video)], str(legacy), width=96, height=64, work_dir=legacy_work,
                         audio_path=str(bed), audio_mode=mode, music_volume=0.6)
    assert bed_alone == pytest.approx(_level(legacy, 0.2, 0.6, 1200), rel=0.06)
    assert _level(out, 0.2, 0.6, 440) == pytest.approx(_level(legacy, 0.2, 0.6, 440),
                                                        abs=0.003)


@NEED_FFMPEG
async def test_short_replace_bed_does_not_shorten_a_composition_with_clips(
    tmp_path: Path,
) -> None:
    video = _video(tmp_path / "v.mp4", 3, frequency=440)
    bed = _tone(tmp_path / "bed.wav", 1200, 0.5)
    voice = _tone(tmp_path / "voice.wav", 1700, 1)
    out = await _compose(tmp_path, [video], [PositionedAudio(str(voice), 2.0)],
                         bed=bed, mode="replace")
    durations = _durations(out)
    assert durations["video"] == pytest.approx(3.0, abs=0.05)
    assert durations["audio"] == pytest.approx(3.0, abs=0.05)
    assert _level(out, 1.0, 0.6, 1200) < 0.002              # bed over: silence
    assert _level(out, 2.2, 0.5, 1700) > 0.05               # the clip is still heard


@NEED_FFMPEG
async def test_default_aac_and_lossless_flac(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.mp4", 2)
    voice = _tone(tmp_path / "voice.wav", 1000, 1)
    default = await _compose(tmp_path, [video], [PositionedAudio(str(voice), 0.5)], name="d")
    assert _codecs(default) == {"h264", "aac"}
    lossless = await _compose(tmp_path, [video], [PositionedAudio(str(voice), 0.5)],
                              policy=LOSSLESS_COMPOSE_RENDER_POLICY, name="l")
    assert _codecs(lossless) == {"ffv1", "flac"}
    assert _durations(lossless)["format"] == pytest.approx(2.0, abs=0.05)


@NEED_FFMPEG
async def test_image_and_timed_video_clips_with_an_audio_track(tmp_path: Path) -> None:
    video = _video(tmp_path / "v.mp4", 2)
    image = tmp_path / "i.png"
    subprocess.run(["convert", "-size", "96x64", "xc:blue", str(image)], check=True,
                   capture_output=True, timeout=30)
    voice = _tone(tmp_path / "voice.wav", 1000, 1)
    out = tmp_path / "out.mp4"
    work = tmp_path / "work"
    work.mkdir()
    await ffmpeg.compose([str(video), str(image)], str(out), width=96, height=64,
                         work_dir=work, clip_windows=[(0.5, 1.5), (0.0, None)],
                         still_durations=[None, 1.0],
                         audio_clips=[PositionedAudio(str(voice), 1.2)])
    durations = _durations(out)
    assert durations["video"] == pytest.approx(2.0, abs=0.05)
    assert durations["audio"] == pytest.approx(2.0, abs=0.05)
    assert _level(out, 1.3, 0.5, 1000) > 0.05


# --- project.export: snapshot, provenance and failures -------------------------


async def _queue_export(client: AsyncClient, pid: str, revision_id: str) -> dict:
    async def leave_queued(_job_id: uuid.UUID) -> None:
        return None

    app.dependency_overrides[get_enqueuer] = lambda: leave_queued
    response = await client.post(f"{API}/jobs", json={
        "capability_id": "project.export", "project_id": pid,
        "input": {"composition_revision_id": revision_id, "preset": "web"},
    })
    assert response.status_code == 201, response.text
    return response.json()


async def _execute(job_id: str) -> Job:
    async with async_session_factory() as session:
        return await execute_job(session, uuid.UUID(job_id))


async def _snapshot(job_id: str) -> dict:
    async with async_session_factory() as session:
        job = await session.get(Job, uuid.UUID(job_id))
        assert job is not None
        return dict(job.input["execution_snapshot"])


async def _project_with_media(client: AsyncClient, make_project, name: str, tmp_path: Path):
    pid = await make_project(name)
    video = await _upload(client, pid, "red.mp4", _video(tmp_path / "v.mp4", 2).read_bytes())
    voice = await _upload(client, pid, "voice.wav",
                          _tone(tmp_path / "voice.wav", 1000, 1).read_bytes())
    return pid, video, voice


@NEED_FFMPEG
async def test_export_snapshot_records_audio_clip_sources(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid, video, voice = await _project_with_media(client, make_project, "audio-export", tmp_path)
    plain = (await _put(client, pid, _document([{"id": _uid(), "asset_id": video}]))).json()
    legacy_job = await _queue_export(client, pid, plain["revision_id"])
    assert "ordered_audio_clip_sources" not in await _snapshot(legacy_job["id"])

    current = await client.get(f"{API}/projects/{pid}/composition")
    response = await client.put(
        f"{API}/projects/{pid}/composition",
        json=_document([{"id": _uid(), "asset_id": video}], [_audio_clip(voice, 0.5)]),
        headers={"If-Match": current.headers["etag"]},
    )
    assert response.status_code == 200, response.text
    job = await _queue_export(client, pid, response.json()["revision_id"])
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(voice))
        assert asset is not None
        checksum = asset.checksum_sha256
    assert (await _snapshot(job["id"]))["ordered_audio_clip_sources"] == [
        {"asset_id": voice, "checksum_sha256": checksum}
    ]
    executed = await _execute(job["id"])
    assert executed.status.value == "succeeded", executed.error
    assert executed.output is not None


@NEED_FFMPEG
async def test_changed_audio_clip_checksum_is_rejected(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid, video, voice = await _project_with_media(client, make_project, "audio-checksum",
                                                  tmp_path)
    saved = (await _put(client, pid, _document(
        [{"id": _uid(), "asset_id": video}], [_audio_clip(voice, 0.5)]
    ))).json()
    job = await _queue_export(client, pid, saved["revision_id"])
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(voice))
        assert asset is not None
        asset.checksum_sha256 = "0" * 64
        await session.commit()
    before = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    executed = await _execute(job["id"])
    assert executed.status.value == "failed"
    assert "does not match the export snapshot" in (executed.error or "")
    after = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    assert after == before


@NEED_FFMPEG
async def test_ai_generated_audio_clip_is_inherited_by_the_export(
    client: AsyncClient, make_project, tmp_path: Path
) -> None:
    pid, video, voice = await _project_with_media(client, make_project, "audio-provenance",
                                                  tmp_path)
    async with async_session_factory() as session:
        asset = await session.get(Asset, uuid.UUID(voice))
        assert asset is not None
        asset.ai_provenance = "generated"  # e.g. audio.tts narration
        await session.commit()
    saved = (await _put(client, pid, _document(
        [{"id": _uid(), "asset_id": video}], [_audio_clip(voice, 0.5)]
    ))).json()
    executed = await _execute((await _queue_export(client, pid, saved["revision_id"]))["id"])
    assert executed.status.value == "succeeded", executed.error
    assert executed.output is not None
    async with async_session_factory() as session:
        output = await session.get(Asset, uuid.UUID(executed.output["outputs"][0]["asset_id"]))
        assert output is not None
        assert output.ai_provenance == "inherited"


@pytest.mark.parametrize("problem", ["corrupt", "starts_after_end"])
@NEED_FFMPEG
async def test_failed_audio_clip_render_creates_no_asset(
    client: AsyncClient, make_project, tmp_path: Path, problem: str
) -> None:
    pid = await make_project(f"audio-fail-{problem}")
    video = await _upload(client, pid, "red.mp4", _video(tmp_path / "v.mp4", 2).read_bytes())
    if problem == "corrupt":
        voice = await _upload(client, pid, "broken.wav", b"not audio")
        clip = _audio_clip(voice, 0.5)
    else:
        voice = await _upload(client, pid, "voice.wav",
                              _tone(tmp_path / "voice.wav", 1000, 1).read_bytes())
        clip = _audio_clip(voice, 5)  # metadata pending: only the render can tell
    saved = await _put(client, pid, _document([{"id": _uid(), "asset_id": video}], [clip]))
    assert saved.status_code == 201, saved.text
    job = await _queue_export(client, pid, saved.json()["revision_id"])
    before = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    executed = await _execute(job["id"])
    assert executed.status.value == "failed" and executed.output is None
    if problem == "starts_after_end":
        assert "must start before the end of the video track" in (executed.error or "")
    after = {a["id"] for a in (await client.get(f"{API}/projects/{pid}/assets")).json()}
    assert after == before
