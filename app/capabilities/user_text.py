"""Plain-language, user-facing capability text.

Written for the editor's target user: a cultural-heritage professional with no
editing background. It is kept apart from ``description`` and parameter
``description`` on purpose. Those stay the precise operational contract that
the Agent planner reads verbatim; this text is exposed only as the additive
``summary`` field and per-parameter ``x-help`` key, which the planner prompt
never includes.

Rules for entries: no tool, library, encoder or capability-ID names; say what
the user gets, not how it is computed; mention a limitation only when it
changes what the user should do.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from app.schemas.capability import CapabilityDefinition


class UserText(TypedDict, total=False):
    summary: str
    params: dict[str, str]


USER_TEXT: dict[str, UserText] = {
    # --- video editing ---------------------------------------------------
    "video.trim": {
        "summary": (
            "Keep one part of a video and discard the rest. Choose where the part "
            "starts and, if you want, where it ends; leave the end empty to keep "
            "everything until the end of the video."
        ),
        "params": {
            "start": "Where the part you want to keep begins, in seconds from the "
                     "start of the video.",
            "end": "Where the part you want to keep ends, in seconds. Leave empty to "
                   "keep everything until the end of the video.",
        },
    },
    "video.split": {
        "summary": (
            "Cut a video into several pieces at the moments you choose. Every piece "
            "is kept and added to your library as a separate video."
        ),
        "params": {
            "markers": "The moments, in seconds, where the video is cut. Two cut "
                       "points make three pieces.",
        },
    },
    "video.concat": {
        "summary": "Join several video clips into one video, in the order you arrange them.",
        "params": {
            "asset_ids": "The clips to join, in the order they should play.",
        },
    },
    "video.transcode": {
        "summary": (
            "Save a copy of a video as a different file type, for example MP4, "
            "which plays almost everywhere. The picture size stays the same, so "
            "the copy is not necessarily smaller; to get a lighter file, also "
            "make the picture smaller with Video Resize."
        ),
        "params": {
            "format": "The file type of the new copy. MP4 plays on almost any "
                      "device and website.",
            "video_codec": "How the picture is stored inside the file. Leave empty "
                           "and a suitable choice is made for the file type.",
            "audio_codec": "How the sound is stored inside the file. Leave empty "
                           "and a suitable choice is made for the file type.",
            "quality": "The balance between picture quality and file size. This "
                       "aims at a level of quality, not a file size: the copy can "
                       "still be larger than the original. Needs a picture "
                       "storage choice above.",
        },
    },
    "video.mute": {
        "summary": "Remove the sound from a video. The picture stays exactly the same.",
    },
    "video.crop": {
        "summary": (
            "Cut away the edges of a video and keep only the area you select. The "
            "area must fit inside the picture; the video's size is shown in its "
            "details."
        ),
        "params": {
            "x": "How far from the left edge the kept area begins, in pixels.",
            "y": "How far from the top edge the kept area begins, in pixels.",
            "width": "How wide the kept area is, in pixels.",
            "height": "How tall the kept area is, in pixels.",
        },
    },
    "video.resize": {
        "summary": (
            "Make the picture of a video larger or smaller. Enter only a height, "
            "for example 720, to keep the picture's proportions."
        ),
        "params": {
            "width": "New width in pixels. Leave empty to keep the proportions.",
            "height": "New height in pixels, for example 480, 720 or 1080. Leave "
                      "empty to keep the proportions.",
        },
    },
    "video.thumbnail": {
        "summary": (
            "Save one still picture from a video at the moment you choose. The "
            "picture is added to your library as an image."
        ),
        "params": {
            "timestamp": "The moment to capture, in seconds from the start of the video.",
        },
    },
    "video.audio.extract": {
        "summary": (
            "Save the sound of a video as a separate audio file. The video itself "
            "is not changed."
        ),
    },
}


def apply_user_text(
    capabilities: Sequence[CapabilityDefinition],
    texts: dict[str, UserText] | None = None,
) -> list[CapabilityDefinition]:
    """Return copies carrying ``summary`` and per-parameter ``x-help``.

    Fails at import time on drift (an entry for an unknown capability or
    parameter), so renamed parameters cannot silently lose their help text.
    """
    texts = USER_TEXT if texts is None else texts
    known = {capability.id for capability in capabilities}
    unknown = set(texts) - known
    if unknown:
        raise RuntimeError(f"User text for unknown capabilities: {sorted(unknown)}")
    result = []
    for capability in capabilities:
        entry = texts.get(capability.id)
        if entry is None:
            result.append(capability)
            continue
        schema = copy.deepcopy(capability.input_schema)
        properties = schema.get("properties", {})
        for name, help_text in entry.get("params", {}).items():
            if name not in properties:
                raise RuntimeError(
                    f"User text for unknown parameter {capability.id}.{name}"
                )
            properties[name]["x-help"] = help_text
        result.append(capability.model_copy(update={
            "summary": entry.get("summary"),
            "input_schema": schema,
        }))
    return result
