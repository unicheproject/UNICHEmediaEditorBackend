"""Which FFmpeg demuxers may open a stored file, and the check every launch passes.

FFmpeg picks a demuxer from the file contents, not the extension. An uploaded
".mp4" that is really an HLS playlist was opened as HLS and FFmpeg followed its
references to other local media, copying them into the output (reproduced
offline, Task #18A). Every FFmpeg/FFprobe input opened from a file therefore
names the allowed demuxers and the file protocol. FFmpeg still picks the best
demuxer for the contents; if that one is not allowed, opening fails.

Allowed demuxers (FFmpeg 5.1 names, measured on fixtures of every accepted
format): mov (mp4, mov, m4v, m4a), matroska (mkv, webm), avi, mp3, wav, flac,
ogg, and the single-image readers png_pipe, jpeg_pipe, webp_pipe, tiff_pipe.
None of them follows references to other files: mov's external tracks stay off
(enable_drefs defaults to false). Playlists (hls, dash), concat and image2
(filename patterns) are not allowed.

Exempt are only inputs whose format is forced (-f) because the application
built or validated them: lavfi sources, the concat lists the application writes
(listing its own intermediate files) and subtitles validated in subtitles.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

MEDIA_DEMUXERS = ("mov", "matroska", "avi", "mp3", "wav", "flac", "ogg")
IMAGE_DEMUXERS = ("png_pipe", "jpeg_pipe", "webp_pipe", "tiff_pipe")
FORMAT_WHITELIST = ",".join(MEDIA_DEMUXERS + IMAGE_DEMUXERS)
# Input options: they must precede the input's -i (ffmpeg) or be global (ffprobe).
UNTRUSTED_INPUT = ("-protocol_whitelist", "file", "-format_whitelist", FORMAT_WHITELIST)
# Forced formats that only application-built or validated inputs use.
TRUSTED_FORCED_FORMATS = frozenset({"lavfi", "concat", "srt", "webvtt"})

# Part of every Agent reuse identity (job nodes and fused render groups). Results
# produced before input confinement may have read other files, so they never
# match identities made under this policy; change it if the policy changes.
MEDIA_INPUT_POLICY = "ffmpeg-input-confinement-v1"

_TOOLS = frozenset({"ffmpeg", "ffprobe"})


class UnrestrictedInputError(ValueError):
    """A command would let FFmpeg choose a demuxer for a file without the allowlist."""


_REQUIRED = {"-protocol_whitelist": "file", "-format_whitelist": FORMAT_WHITELIST}
_POLICY_OPTIONS = ("-f", *_REQUIRED)


def _restricted(options: Sequence[str]) -> bool:
    """Whether one input's options restrict it, judged by their effective values.

    Either a single trusted forced format with no policy options, or each
    policy option exactly once with its required value and no -f. Repeated or
    conflicting policy options (FFmpeg uses the last one), any other forced
    format and an option without a value are refused (fail closed).
    """
    values: dict[str, list[str]] = {option: [] for option in _POLICY_OPTIONS}
    for index, token in enumerate(options):
        if token in values:
            if index + 1 >= len(options):
                return False
            values[token].append(options[index + 1])
    if values["-f"]:
        return (
            len(values["-f"]) == 1 and values["-f"][0] in TRUSTED_FORCED_FORMATS
            and not any(values[option] for option in _REQUIRED)
        )
    return all(values[option] == [value] for option, value in _REQUIRED.items())


def check_input_policy(args: Sequence[str]) -> None:
    """Refuse FFmpeg/FFprobe commands with an input that lacks the allowlist.

    ffmpeg: each input's own options (the tokens after the previous input,
    up to its -i) must restrict it (see _restricted).
    ffprobe: exactly one input, given with -i (no positional input); all other
    tokens are its options.
    """
    if not args or Path(args[0]).name not in _TOOLS:
        return
    tokens = list(args)
    if Path(tokens[0]).name == "ffprobe":
        if tokens.count("-i") != 1 or tokens.index("-i") == len(tokens) - 1:
            raise UnrestrictedInputError("ffprobe input is not restricted")
        at = tokens.index("-i")
        if not _restricted(tokens[1:at] + tokens[at + 2:]):
            raise UnrestrictedInputError("ffprobe input is not restricted")
        return
    start = 1
    for index, token in enumerate(tokens):
        if token == "-i":
            if index == len(tokens) - 1 or not _restricted(tokens[start:index]):
                raise UnrestrictedInputError("ffmpeg input is not restricted")
            start = index + 2
