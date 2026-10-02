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
}
