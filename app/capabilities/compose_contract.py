"""Shared semantic defaults for the direct and saved composition contracts."""

from __future__ import annotations

from enum import StrEnum


class ComposeAudioMode(StrEnum):
    MIX = "mix"
    DUCK = "duck"
    REPLACE = "replace"


COMPOSE_DEFAULT_AUDIO_MODE = ComposeAudioMode.MIX
COMPOSE_DEFAULT_MUSIC_VOLUME = 0.3

# Still-image clips in saved compositions. The duration bounds are one output
# frame (compositions render at 30 fps) and an initial resource-protection cap;
# the pixel cap bounds the decoded size of a source image.
IMAGE_CLIP_MIN_DURATION_SECONDS = 1 / 30
IMAGE_CLIP_MAX_DURATION_SECONDS = 600.0
IMAGE_CLIP_MAX_PIXELS = 100_000_000
