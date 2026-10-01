"""Typed, conservative compiler for the Agent's first linear video render set.

This is deliberately Agent-only. Direct capability jobs continue to use their
existing handlers and FFmpeg calls.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.agent.schemas import Plan, PlanStep, is_step_ref, step_ref_id

OPTIMIZER_VERSION = "agent-video-render-v1"
_ADAPTERS = {"video.trim", "video.resize", "video.crop", "video.lowerthird"}
_SUPPORTED_CONTAINERS = {"mp4", "m4v", "mov", "mkv", "webm", "avi"}
_ADAPTER_PARAMS = {
    "video.trim": {"start", "end"},
    "video.resize": {"width", "height"},
    "video.crop": {"x", "y", "width", "height"},
    "video.lowerthird": {
        "text", "caption", "start", "duration", "position", "background", "foreground"
    },
}


@dataclass(frozen=True)
class TemporalTrim:
    start: float
    end: float


@dataclass(frozen=True)
class Resize:
    width: int
    height: int


@dataclass(frozen=True)
class Crop:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class Lowerthird:
    text: str
    caption: str | None
    start: float
    duration: float
    position: str
    background: str
    foreground: str


RenderOperation = TemporalTrim | Resize | Crop | Lowerthird


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def plan_fingerprint(plan: Plan) -> str:
    return _canonical_hash([step.model_dump(mode="json") for step in plan.steps])


def _adapter_eligible(step: PlanStep) -> bool:
    return (
        step.capability_id in _ADAPTERS
        and set(step.params).issubset(_ADAPTER_PARAMS[step.capability_id])
        and not step.assets
    )


def _video_settings(extension: str) -> list[str]:
    if extension in {"mp4", "m4v", "mov", "mkv"}:
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p"]
    if extension == "webm":
        return ["-c:v", "libvpx-vp9", "-crf", "31", "-b:v", "0", "-pix_fmt", "yuv420p"]
    if extension == "avi":
        return ["-c:v", "mpeg4", "-q:v", "5", "-pix_fmt", "yuv420p"]
    return []


def _operation(step: PlanStep) -> dict[str, Any]:
    """Translate only declared adapter parameters into a typed operation."""
    params = dict(step.params)
    if step.capability_id == "video.trim":
        value: RenderOperation = TemporalTrim(params["start"], params["end"])
    elif step.capability_id == "video.resize":
        value = Resize(params["width"], params["height"])
    elif step.capability_id == "video.crop":
        value = Crop(params["x"], params["y"], params["width"], params["height"])
    elif step.capability_id == "video.lowerthird":
        value = Lowerthird(
            text=params["text"], caption=params.get("caption"),
            start=params.get("start", 0), duration=params.get("duration", 5),
            position=params.get("position", "bottom-left"),
            background=params.get("background", "#101418"),
            foreground=params.get("foreground", "#ffffff"),
        )
    else:
        raise ValueError(f"No Agent render adapter for {step.capability_id}")
    return {"kind": type(value).__name__.lower(), "value": value.__dict__}


def _is_identity_crop(op: dict[str, Any], width: int | None, height: int | None) -> bool:
    if op["kind"] != "crop" or width is None or height is None:
        return False
    value = op["value"]
    return (
        value["x"] == 0 and value["y"] == 0
        and value["width"] == width and value["height"] == height
    )


def compile_execution_plan(plan: Plan, asset_facts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return deterministic physical nodes and truthful logical-step mappings.

    A fused group starts only from an in-scope literal video Asset whose
    container is known to be supported. Subsequent operations must consume the
    immediately preceding step and that output must have exactly one consumer.
    Anything else remains a normal per-capability Job boundary.
    """
    consumers: dict[str, int] = {step.id: 0 for step in plan.steps}
    def count_refs(value: Any) -> None:
        if is_step_ref(value):
            ref = step_ref_id(value)
            if ref in consumers:
                consumers[ref] += 1
        elif isinstance(value, list):
            for item in value:
                count_refs(item)
        elif isinstance(value, dict):
            for item in value.values():
                count_refs(item)

    for step in plan.steps:
        for value in [step.asset, *step.assets, *step.params.values()]:
            count_refs(value)

    nodes: list[dict[str, Any]] = []
    step_mapping: dict[str, dict[str, Any]] = {}
    i = 0
    group_number = 0
    video_encodes = 0
    audio_encodes = 0
    audio_count_known = True
    while i < len(plan.steps):
        step = plan.steps[i]
        facts = asset_facts.get(step.asset or "", {})
        ext = str(facts.get("extension") or "").lower().lstrip(".")
        can_start = (
            _adapter_eligible(step)
            and step.asset is not None and not is_step_ref(step.asset)
            and not step.depends_on
            and not any(key.endswith("_asset_id") for key in step.params)
            and facts.get("media_type") == "video"
            and ext in _SUPPORTED_CONTAINERS
        )
        if not can_start:
            nodes.append({"kind": "job", "step_id": step.id})
            step_mapping[step.id] = {"step_id": step.id, "kind": "job", "status": "planned"}
            i += 1
            continue

        group_number += 1
        group_id = f"render-group-{group_number}"
        root_id = str(step.asset)
        metadata_available = facts.get("technical_metadata_status") == "available"
        width = facts.get("width") if metadata_available else None
        height = facts.get("height") if metadata_available else None
        if not isinstance(width, int) or width <= 0:
            width = None
        if not isinstance(height, int) or height <= 0:
            height = None
        group_ops: list[dict[str, Any]] = []
        group_steps: list[str] = []
        source_checksum = facts.get("checksum_sha256")
        output_ext = ext

        while i < len(plan.steps):
            candidate = plan.steps[i]
            if not _adapter_eligible(candidate):
                break
            if any(key.endswith("_asset_id") for key in candidate.params):
                break
            if group_steps:
                prev = plan.steps[i - 1]
                if candidate.asset != f"@{prev.id}" or consumers.get(prev.id) != 1:
                    break
                if candidate.depends_on and set(candidate.depends_on) - {prev.id}:
                    break
            if any(value != (group_steps[-1] if group_steps else "")
                   for value in candidate.depends_on):
                break
            if not group_steps and candidate.asset != root_id:
                break

            op = _operation(candidate)
            identity = _is_identity_crop(op, width, height)
            group_ops.append({
                "step_id": candidate.id,
                "capability_id": candidate.capability_id,
                "params": candidate.params,
                "operation": op,
                "identity": identity,
            })
            group_steps.append(candidate.id)
            if not identity:
                value = op["value"]
                if op["kind"] == "resize":
                    width = value["width"] if value["width"] > 0 else None
                    height = value["height"] if value["height"] > 0 else None
                elif op["kind"] == "crop":
                    width, height = value["width"], value["height"]
                elif op["kind"] == "lowerthird":
                    output_ext = "mp4"
            step_mapping[candidate.id] = {
                "step_id": candidate.id,
                "kind": "render_group_step",
                "render_group_id": group_id,
                "identity": identity,
                "status": "planned",
            }
            i += 1
            if consumers.get(candidate.id, 0) > 1:
                break

        active_ops = [op for op in group_ops if not op["identity"]]
        if active_ops:
            has_trim = any(op["operation"]["kind"] == "temporaltrim" for op in active_ops)
            known_audio = facts.get("has_audio_stream")
            audio_policy = (
                "encode-container-compatible-audio" if has_trim and known_audio is True
                else "copy-audio" if not has_trim and output_ext == ext
                else "conditional-audio-policy"
            )
            group_fingerprint = _canonical_hash({
                "version": OPTIMIZER_VERSION,
                "source_checksum": source_checksum,
                "source_asset_id": root_id,
                "operations": group_ops,
                "output_extension": output_ext,
                "video_settings": _video_settings(output_ext),
                "audio_policy": audio_policy,
            })
            nodes.append({
                "kind": "render_group", "id": group_id, "step_ids": group_steps,
                "source_asset_id": root_id, "source_checksum": source_checksum,
                "source_metadata_fingerprint": _canonical_hash({
                    "status": facts.get("technical_metadata_status"),
                    "width": facts.get("width"), "height": facts.get("height"),
                    "has_audio_stream": facts.get("has_audio_stream"),
                }),
                "source_metadata": {
                    "status": facts.get("technical_metadata_status"),
                    "width": facts.get("width"), "height": facts.get("height"),
                    "has_audio_stream": facts.get("has_audio_stream"),
                },
                "source_extension": ext, "output_extension": output_ext,
                "operations": group_ops, "identity_hash": group_fingerprint,
                "status": "planned", "job_id": None, "output_asset_id": None,
            })
            video_encodes += 1
            audio_reencode = has_trim or output_ext != ext
            if audio_reencode:
                if facts.get("has_audio_stream") is True:
                    audio_encodes += 1
                elif facts.get("has_audio_stream") is None:
                    audio_count_known = False
        else:
            nodes.append({
                "kind": "identity", "id": group_id, "step_ids": group_steps,
                "source_asset_id": root_id,
                "source_checksum": source_checksum,
                "source_metadata_fingerprint": _canonical_hash({
                    "status": facts.get("technical_metadata_status"),
                    "width": facts.get("width"), "height": facts.get("height"),
                    "has_audio_stream": facts.get("has_audio_stream"),
                }),
                "status": "planned",
                "output_asset_id": root_id,
            })

    opaque_jobs = any(node["kind"] == "job" for node in nodes)
    summary = {
        "optimizer_version": OPTIMIZER_VERSION,
        "input_fingerprint": plan_fingerprint(plan),
        "render_group_count": sum(node["kind"] == "render_group" for node in nodes),
        "video_encode_count": video_encodes if not opaque_jobs else None,
        "audio_encode_count": audio_encodes if audio_count_known and not opaque_jobs else None,
        "encode_counts_complete": not opaque_jobs and audio_count_known,
        "nodes": nodes,
        "steps": [step_mapping[s.id] for s in plan.steps],
    }
    return summary


def typed_operations(group: dict[str, Any]) -> list[RenderOperation | None]:
    """Materialize the serializable compiled operations as typed values."""
    result: list[RenderOperation | None] = []
    for entry in group["operations"]:
        if entry["identity"]:
            result.append(None)
            continue
        op, value = entry["operation"]["kind"], entry["operation"]["value"]
        if op == "temporaltrim":
            result.append(TemporalTrim(**value))
        elif op == "resize":
            result.append(Resize(**value))
        elif op == "crop":
            result.append(Crop(**value))
        elif op == "lowerthird":
            result.append(Lowerthird(**value))
        else:  # pragma: no cover - compiler controls this serialized enum
            raise ValueError(f"Unknown compiled render operation: {op}")
    return result


async def render_group(source: Path, output: Path, work_dir: Path, group: dict[str, Any]) -> bool:
    """Render a group with one explicit FFmpeg invocation; return audio presence.

    Selects first video/audio streams and intentionally omits subtitle/data and
    additional tracks, following the backend's current first-stream transform
    convention.
    """
    from app.core.errors import ToolError
    from app.tools import ffmpeg, imagemagick, runner

    operations = typed_operations(group)
    active = [op for op in operations if op is not None]
    if not active:
        raise ValueError("An identity-only render group must not be rendered")
    audio_present = await ffmpeg._has_audio(str(source))
    output_ext = group["output_extension"]
    video_args = _video_settings(output_ext)
    if not video_args:
        raise ToolError(f"Agent render does not support the '{output_ext}' container")

    args = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source)]
    graph: list[str] = []
    current = "0:v:0"
    audio_current = "0:a:0"
    video_index = 0
    audio_index = 0
    temporal = False
    container_change = output_ext != group["source_extension"]
    overlay_index = 1
    for operation in operations:
        if operation is None:
            continue
        if isinstance(operation, TemporalTrim):
            temporal = True
            video_index += 1
            next_video = f"v{video_index}"
            graph.append(
                f"[{current}]trim=start={operation.start}:end={operation.end},setpts=PTS-STARTPTS[{next_video}]"
            )
            current = next_video
            if audio_present:
                audio_index += 1
                next_audio = f"a{audio_index}"
                graph.append(
                    f"[{audio_current}]atrim=start={operation.start}:end={operation.end},asetpts=PTS-STARTPTS[{next_audio}]"
                )
                audio_current = next_audio
        elif isinstance(operation, Resize):
            video_index += 1
            next_video = f"v{video_index}"
            graph.append(f"[{current}]scale={operation.width}:{operation.height}[{next_video}]")
            current = next_video
        elif isinstance(operation, Crop):
            video_index += 1
            next_video = f"v{video_index}"
            graph.append(
                f"[{current}]crop={operation.width}:{operation.height}:{operation.x}:{operation.y}[{next_video}]"
            )
            current = next_video
        elif isinstance(operation, Lowerthird):
            overlay_path = work_dir / f"lowerthird-{overlay_index}.png"
            await imagemagick.render_lowerthird(
                str(overlay_path), operation.text, operation.caption,
                background=operation.background, foreground=operation.foreground,
            )
            args += ["-loop", "1", "-i", str(overlay_path)]
            xy = {
                "bottom-left": "20:main_h-overlay_h-20",
                "bottom-center": "(main_w-overlay_w)/2:main_h-overlay_h-20",
                "bottom-right": "main_w-overlay_w-20:main_h-overlay_h-20",
            }.get(operation.position)
            if xy is None:
                raise ToolError("Unsupported lower-third position")
            video_index += 1
            next_video = f"v{video_index}"
            graph.append(
                f"[{current}][{overlay_index}:v]overlay={xy}:shortest=1:"
                f"enable='between(t,{operation.start},{operation.start + operation.duration})'"
                f"[{next_video}]"
            )
            current = next_video
            overlay_index += 1

    args += ["-filter_complex", ";".join(graph), "-map", f"[{current}]"]
    args += video_args
    if audio_present:
        if temporal:
            audio_codec = {
                "mp4": ("aac", "192k"), "m4v": ("aac", "192k"), "mov": ("aac", "192k"),
                "mkv": ("aac", "192k"), "webm": ("libopus", "128k"),
                "avi": ("libmp3lame", "192k"),
            }.get(output_ext)
            if audio_codec is None:
                raise ToolError(f"Cannot preserve trimmed audio in '{output_ext}'")
            args += ["-map", f"[{audio_current}]", "-c:a", audio_codec[0], "-b:a", audio_codec[1]]
        elif container_change:
            args += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "192k"]
        else:
            args += ["-map", "0:a:0?", "-c:a", "copy"]
    else:
        args += ["-an"]
    args.append(str(output))
    await runner.run(args)
    return audio_present
