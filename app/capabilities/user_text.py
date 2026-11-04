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
    # --- image editing ---------------------------------------------------
    "image.resize": {
        "summary": (
            "Make an image larger or smaller. The image gets exactly the width and "
            "height you enter, so if they do not match its proportions the picture "
            "will look stretched or squashed."
        ),
        "params": {
            "width": "New width in pixels.",
            "height": "New height in pixels.",
        },
    },
    "image.crop": {
        "summary": (
            "Cut away the edges of an image and keep only the area you select. The "
            "image's size is shown in its details."
        ),
        "params": {
            "x": "How far from the left edge the kept area begins, in pixels.",
            "y": "How far from the top edge the kept area begins, in pixels.",
            "width": "How wide the kept area is, in pixels.",
            "height": "How tall the kept area is, in pixels.",
        },
    },
    "image.format": {
        "summary": (
            "Save a copy of an image as a different file type. JPEG and WebP make "
            "small files for the web; PNG and TIFF keep every detail and suit "
            "archiving."
        ),
        "params": {
            "format": "The file type of the new copy.",
        },
    },
    "image.colour.adjust": {
        "summary": (
            "Make an image brighter or darker, change its contrast, or make its "
            "colours stronger or weaker. Values you leave unchanged have no effect."
        ),
        "params": {
            "brightness": "100 keeps the image as it is. Lower values make it "
                          "darker, higher values make it brighter.",
            "contrast": "0 keeps the image as it is. Negative values make it flatter, "
                        "positive values make light and dark areas stand out more.",
            "saturation": "100 keeps the colours as they are. 0 makes the image "
                          "black and white; higher values make colours stronger.",
        },
    },
    "image.upscale": {
        "summary": (
            "Make an image larger while recreating fine detail, so it stays sharp "
            "instead of looking blurry. Works best on photographs; very small, grainy "
            "or black-and-white images can come out with invented details, so check "
            "the result."
        ),
        "params": {
            "scale": "How many times larger the image becomes. 4 means four times "
                     "the width and four times the height.",
        },
    },
    # --- audio -----------------------------------------------------------
    "audio.trim": {
        "summary": "Keep one part of an audio recording and discard the rest.",
        "params": {
            "start": "Where the part you want to keep begins, in seconds from the "
                     "start of the recording.",
            "end": "Where the part you want to keep ends, in seconds from the start "
                   "of the recording.",
        },
    },
    "audio.concat": {
        "summary": "Join several audio recordings into one, in the order you arrange them.",
        "params": {
            "asset_ids": "The recordings to join, in the order they should play.",
        },
    },
    "audio.gain": {
        "summary": (
            "Make the sound louder or quieter. Works on audio files and on the "
            "sound of a video; a video keeps its picture unchanged."
        ),
        "params": {
            "gain_db": "How much louder or quieter, in decibels. 0 keeps it as it "
                       "is, +6 is clearly louder, -6 is clearly quieter. Too much "
                       "increase can make loud passages sound distorted.",
        },
    },
    "audio.normalize": {
        "summary": (
            "Bring the sound to a standard overall loudness, so recordings play at "
            "a similar volume. Works on audio files and on the sound of a video."
        ),
        "params": {
            "target_i": "How loud the result should be overall. -16 suits websites "
                        "and podcasts; -23 is the European broadcast standard. Keep "
                        "the default if unsure.",
        },
    },
    "audio.fade": {
        "summary": (
            "Let the sound rise gently at the start and/or fade out at the end. "
            "Works on audio files and on the sound of a video."
        ),
        "params": {
            "fade_in": "How long the sound takes to rise at the start, in seconds. "
                       "0 means no fade-in.",
            "fade_out": "How long the sound takes to fade out at the end, in "
                        "seconds. 0 means no fade-out.",
        },
    },
    "audio.transcode": {
        "summary": (
            "Save a copy of an audio recording as a different file type. MP3 and "
            "M4A make small files for the web; WAV and FLAC keep the full quality "
            "and suit archiving."
        ),
        "params": {
            "format": "The file type of the new copy.",
            "codec": "How the sound is stored inside the file. Leave empty and a "
                     "suitable choice is made for the file type.",
        },
    },
    "audio.denoise": {
        "summary": (
            "Reduce steady background noise, such as hum or hiss, behind speech. "
            "Works on audio files and on the sound of a video."
        ),
        "params": {
            "strength": "1 removes as much noise as possible, 0.5 about half, 0 "
                        "leaves the sound unchanged. Negative values do the "
                        "opposite and keep mostly the noise.",
        },
    },
    "audio.mix": {
        "summary": (
            "Add background music or narration to a video or audio recording. You "
            "can play both together, keep the added track quietly in the "
            "background, or replace the original sound."
        ),
        "params": {
            "music_asset_id": "The music or narration recording to add.",
            "music_volume": "How loud the added recording is. 1 is its own volume, "
                            "0.3 is a quiet background, 0 is silent.",
            "mode": "Mix plays both together; Duck keeps the added recording quiet "
                    "underneath; Replace removes the original sound.",
        },
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
