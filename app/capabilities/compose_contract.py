"""Shared semantic defaults for the direct and saved composition contracts."""

from __future__ import annotations

from enum import StrEnum


class ComposeAudioMode(StrEnum):
    MIX = "mix"
    DUCK = "duck"
    REPLACE = "replace"


COMPOSE_DEFAULT_AUDIO_MODE = ComposeAudioMode.MIX
COMPOSE_DEFAULT_MUSIC_VOLUME = 0.3
