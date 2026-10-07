"""Portable, validated media option domains shared by capabilities and export."""

VIDEO_TRANSCODE_OPTIONS: dict[str, dict[str, list[str]]] = {
    "mp4": {"video_codec": ["libx264"], "audio_codec": ["aac"]},
    "mov": {"video_codec": ["libx264"], "audio_codec": ["aac"]},
    "mkv": {
        "video_codec": ["libx264", "libx265", "libvpx-vp9"],
        "audio_codec": ["aac", "libmp3lame", "libopus", "libvorbis", "flac"],
    },
    "webm": {"video_codec": ["libvpx-vp9"], "audio_codec": ["libopus", "libvorbis"]},
}

VIDEO_CODEC_DESCRIPTIONS = {
    "libx264": "H.264 / AVC",
    "libx265": "H.265 / HEVC",
    "libvpx-vp9": "VP9",
}

AUDIO_CODEC_DESCRIPTIONS = {
    "aac": "AAC",
    "libmp3lame": "MP3",
    "libopus": "Opus",
    "libvorbis": "Vorbis",
    "flac": "FLAC",
    "pcm_s16le": "Uncompressed signed 16-bit PCM",
}

# Maintained audio-only choices, matching the supported audio Asset families.
# Codec omission retains FFmpeg's container default; explicit choices are
# constrained by this same matrix in discovery, admission and the handler.
AUDIO_TRANSCODE_OPTIONS: dict[str, dict[str, list[str]]] = {
    "mp3": {"codec": ["libmp3lame"]},
    "wav": {"codec": ["pcm_s16le"]},
    "m4a": {"codec": ["aac"]},
    "flac": {"codec": ["flac"]},
    "ogg": {"codec": ["libvorbis", "libopus"]},
}

AUDIO_CONTAINER_DESCRIPTIONS = {
    "mp3": "MP3 audio",
    "wav": "WAV / uncompressed PCM audio",
    "m4a": "MPEG-4 audio / AAC",
    "flac": "FLAC / lossless audio",
    "ogg": "Ogg / Vorbis or Opus audio",
}
