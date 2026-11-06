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
            "quality": "The balance between picture quality and file size. High "
                       "and Balanced aim at a level of quality, not a file size, so "
                       "the copy can be larger than the original. Compressed always "
                       "keeps the picture smaller than the original's. Needs a "
                       "picture storage choice above.",
        },
    },
    "video.export": {
        "summary": (
            "Make a copy of a video ready for where it will be used: the web, social "
            "media, a quick preview, or long-term archiving. Web and Preview copies "
            "are usually lighter, but a video that is already well compressed can "
            "come out larger; the result shows how its size compares. The Archival "
            "copy keeps every detail and is much larger."
        ),
        "params": {
            "preset": "Web plays on any website; Social makes a square or upright "
                      "video for social media; Preview is a small file for checking; "
                      "Archival keeps full quality for preservation.",
            "social_format": "Square for most feeds, or upright (portrait) for "
                             "phone-first platforms. Required for Social.",
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
            "Make an image larger or smaller. Enter only a width or only a height to "
            "keep the picture's proportions. If you enter both, the image gets exactly "
            "that size and may look stretched or squashed."
        ),
        "params": {
            "width": "New width in pixels. Leave empty to keep the proportions.",
            "height": "New height in pixels. Leave empty to keep the proportions.",
        },
    },
    "image.crop": {
        "summary": (
            "Cut away the edges of an image and keep only the area you select. The "
            "area must fit inside the picture; the image's size is shown in its "
            "details."
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
            "quality": "For JPEG and WebP only: how much detail to keep. Higher keeps "
                       "more detail and makes a larger file; 90 is very good, 75 is fine "
                       "for the web. Leave empty for a high-quality default.",
            "lossless": "For WebP only: keep every detail exactly, like PNG, at a "
                        "larger file size.",
            "flatten_transparency": "Fill see-through areas with white. JPEG always "
                                    "does this, because it cannot store transparency.",
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
        "summary": (
            "Keep one part of an audio recording and discard the rest. Leave the end "
            "empty to keep everything until the end of the recording."
        ),
        "params": {
            "start": "Where the part you want to keep begins, in seconds from the "
                     "start of the recording.",
            "end": "Where the part you want to keep ends, in seconds. Leave empty to "
                   "keep everything until the end of the recording.",
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
            "a similar volume. Works on audio files and on the sound of a video. "
            "The file keeps its original sound quality settings, so its size stays "
            "about the same."
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
    # --- composition and export -----------------------------------------
    "image.slideshow": {
        "summary": (
            "Turn a set of images into a video slideshow, showing each image for "
            "the same length of time, in the order you arrange them."
        ),
        "params": {
            "asset_ids": "The images to show, in the order they should appear.",
            "seconds_per_image": "How long each image stays on screen, in seconds.",
            "width": "Width of the video in pixels. 1920 with a height of 1080 is "
                     "Full HD.",
            "height": "Height of the video in pixels. 1080 with a width of 1920 is "
                      "Full HD.",
        },
    },
    "image.kenburns": {
        "summary": (
            "Turn a still image into a short video with a slow zoom and pan, "
            "a classic documentary effect. The movement is aimed automatically at "
            "the most important part of the picture, such as a face."
        ),
        "params": {
            "duration": "Length of the video, in seconds.",
            "zoom": "How far the camera moves in. 1.15 is a gentle zoom; 1.5 is "
                    "strong. Must be more than 1.",
            "direction": "In moves closer to the picture over time; Out starts close "
                         "and moves away.",
            "width": "Width of the video in pixels.",
            "height": "Height of the video in pixels.",
        },
    },
    "media.titlecard": {
        "summary": (
            "Create a short video showing a line of text, for use as an opening "
            "title or a chapter heading in a longer video."
        ),
        "params": {
            "text": "The text shown on the title card.",
            "duration": "How long the title card lasts, in seconds.",
            "width": "Width of the video in pixels; match the video it will join.",
            "height": "Height of the video in pixels; match the video it will join.",
            "background": "Background colour, as a name such as black or a code "
                          "such as #101418.",
            "foreground": "Text colour, as a name such as white or a code such as "
                          "#ffffff.",
        },
    },
    "video.lowerthird": {
        "summary": (
            "Show a name and an optional caption in a banner near the bottom of a "
            "video, for example to introduce a speaker, for the time you choose."
        ),
        "params": {
            "text": "The main line, usually a name.",
            "caption": "An optional second line, such as a role or place.",
            "start": "When the banner appears, in seconds from the start of the video.",
            "duration": "How long the banner stays visible, in seconds.",
            "position": "Where along the bottom of the picture the banner appears.",
            "background": "Banner colour, as a name such as black or a code such as "
                          "#101418.",
            "foreground": "Text colour, as a name such as white or a code such as "
                          "#ffffff.",
        },
    },
    "video.subtitle.embed": {
        "summary": (
            "Add subtitles from a subtitle file (SRT or VTT) to a video. They can "
            "be a track viewers switch on and off, or be printed permanently into "
            "the picture."
        ),
        "params": {
            "subtitle_asset_id": "The subtitle file to add.",
            "mode": "Soft adds subtitles viewers can switch on and off in their "
                    "player; Burn prints them into the picture so they always show, "
                    "on any player or website.",
        },
    },
    "video.compose": {
        "summary": (
            "Put several video clips together into one finished video at a single "
            "picture size, with optional background music or narration and "
            "optional subtitles printed into the picture."
        ),
        "params": {
            "asset_ids": "The clips, in the order they should play.",
            "width": "Width of the finished video in pixels. Clips of other sizes "
                     "are fitted to it.",
            "height": "Height of the finished video in pixels. Clips of other sizes "
                      "are fitted to it.",
            "audio_asset_id": "Optional music or narration to add to the whole video.",
            "audio_mode": "Mix plays it together with the clips' own sound; Duck "
                          "keeps it quiet underneath; Replace removes the clips' "
                          "own sound.",
            "music_volume": "How loud the added music or narration is. 1 is its own "
                            "volume, 0.3 is a quiet background.",
            "subtitle_asset_id": "Optional subtitle file to print into the picture.",
        },
    },
    "project.export": {
        "summary": (
            "Produce the final file of a saved project, prepared for where it will "
            "be used: the web, social media, a quick preview, or long-term "
            "archiving."
        ),
        "params": {
            "composition_revision_id": "The saved version of the project to export.",
            "preset": "Web plays on any website; Social makes a square or upright "
                      "video for social media; Preview is a small file for checking; "
                      "Archival keeps full quality for preservation (very large "
                      "files); Advanced lets you set the details yourself.",
            "social_format": "Square for most feeds, or upright (portrait) for "
                             "phone-first platforms. Required for Social.",
            "format": "File type for Advanced exports. MP4 plays almost everywhere.",
            "video_codec": "How the picture is stored, for Advanced exports.",
            "audio_codec": "How the sound is stored, for Advanced exports.",
            "max_width": "Optional largest width in pixels. Smaller videos are never "
                         "enlarged.",
            "max_height": "Optional largest height in pixels. Smaller videos are "
                          "never enlarged.",
            "crf": "Picture quality for Advanced exports: lower numbers mean better "
                   "quality and larger files. 22 is a good balance.",
            "audio_bitrate_kbps": "Sound quality for Advanced exports: higher means "
                                  "better sound and larger files. 128 suits most uses.",
        },
    },
    # --- AI-assisted -----------------------------------------------------
    # Summaries describe what the operation does. Whether it can run on this
    # installation is reported separately by readiness/availability, so these
    # texts must not say "not available".
    "audio.transcribe": {
        "summary": (
            "Write out the words spoken in an audio recording or video as text. "
            "The result is produced by AI, so check names and specialist terms."
        ),
        "params": {
            "asset_id": "The recording or video whose speech should be written out.",
        },
    },
    "subtitle.autogenerate": {
        "summary": (
            "Create subtitles with timings automatically from the speech in a "
            "recording or video, ready to add to the video. Produced by AI; check "
            "them before publishing."
        ),
        "params": {
            "asset_id": "The recording or video to create subtitles for.",
            "language": "The language spoken, as a two-letter code such as en or el. "
                        "Leave empty to detect it automatically.",
        },
    },
    "audio.tts": {
        "summary": (
            "Turn written text into spoken narration with an AI voice, added to "
            "your library as an audio recording."
        ),
        "params": {
            "text": "The text to be read aloud.",
            "voice": "Optional voice to use. Leave empty for the default voice.",
        },
    },
    "audio.music.generate": {
        "summary": (
            "Create original background music with AI from a short description of "
            "the mood or style you want."
        ),
        "params": {
            "prompt": "Describe the music, for example 'calm piano, slow, reflective'.",
        },
    },
    "audio.separate.stems": {
        "summary": (
            "Split a recording into separate parts, such as voice and music, so "
            "each can be used or adjusted on its own. Done with AI."
        ),
        "params": {
            "asset_id": "The recording to split into parts.",
        },
    },
    "image.background.remove": {
        "summary": (
            "Remove the background of an image and keep only the main subject. "
            "Done with AI."
        ),
        "params": {
            "asset_id": "The image whose background should be removed.",
        },
    },
    "image.inpaint": {
        "summary": (
            "Fill in or replace a marked area of an image, for example to remove "
            "a scratch or an unwanted object. Done with AI, which invents the new "
            "content; label results accordingly."
        ),
        "params": {
            "asset_id": "The image to change.",
            "mask_url": "The marked area to fill in or replace.",
            "prompt": "Describe what should appear in the marked area.",
        },
    },
    "image.restore.face": {
        "summary": (
            "Make faces in an old or blurry image clearer. Done with AI, which can "
            "change how a face looks; compare with the original."
        ),
        "params": {
            "asset_id": "The image with faces to restore.",
        },
    },
    "image.caption": {
        "summary": (
            "Write a short description of what an image shows, useful as alt text "
            "or a starting point for a caption. Produced by AI; check it before use."
        ),
        "params": {
            "asset_id": "The image to describe.",
        },
    },
    "image.tag": {
        "summary": (
            "Suggest keywords describing what an image shows, to help organise and "
            "find it later. Produced by AI; review them before use."
        ),
        "params": {
            "asset_id": "The image to suggest keywords for.",
        },
    },
    "video.upscale": {
        "summary": (
            "Make the picture of a video larger while recreating fine detail, so "
            "it stays sharp instead of looking blurry. Done with AI."
        ),
        "params": {
            "asset_id": "The video to enlarge.",
        },
    },
    "video.shot.detect": {
        "summary": (
            "Find the points in a video where one shot or scene changes to the next, "
            "and list their times. Nothing is cut; use the times to split or trim "
            "the video afterwards."
        ),
        "params": {
            "threshold": "How big a change counts as a new shot. Lower finds more "
                         "changes, including small ones; higher finds only clear cuts. "
                         "Keep the default if unsure.",
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
