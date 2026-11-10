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

VIDEO_QUALITY_DESCRIPTIONS = {
    "high": "Prioritize visual quality; usually produces larger files.",
    "balanced": "Balance visual quality and compression.",
    "compressed": "Prioritize smaller files at lower visual quality; when the source "
                  "video bitrate is known, the video bitrate is capped at 75% of it. "
                  "Audio is not capped, so the total file is not guaranteed to shrink.",
}

# These are codec-specific product choices, not equivalent numeric quality scales.
# H.264 follows the existing export CRF range/web/preview policy. HEVC and VP9
# are centered on their encoder defaults (28 and 32), verified in the runtime.
# Only explicit quality selections use these values; omission emits no new options.
VIDEO_QUALITY_CRF: dict[str, dict[str, int]] = {
    "libx264": {"high": 18, "balanced": 22, "compressed": 28},
    "libx265": {"high": 22, "balanced": 28, "compressed": 34},
    "libvpx-vp9": {"high": 24, "balanced": 32, "compressed": 40},
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


# Burned-in subtitle styling (video.subtitle.embed, mode="burn"). Font IDs map to
# libass FontName/Bold pairs that fontconfig resolves to the bundled DejaVu files
# (verified: "DejaVu Serif" + Bold=1 -> DejaVuSerif-Bold.ttf). Sizes and margins
# are libass script units, scaled with the video height.
SUBTITLE_FONTS: dict[str, tuple[str, int]] = {
    "dejavu-sans": ("DejaVu Sans", 0),
    "dejavu-sans-bold": ("DejaVu Sans", 1),
    "dejavu-serif": ("DejaVu Serif", 0),
    "dejavu-serif-bold": ("DejaVu Serif", 1),
}
SUBTITLE_FONT_DESCRIPTIONS = {
    "dejavu-sans": "DejaVu Sans, regular.",
    "dejavu-sans-bold": "DejaVu Sans, bold.",
    "dejavu-serif": "DejaVu Serif, regular.",
    "dejavu-serif-bold": "DejaVu Serif, bold.",
}
# force_style reads Alignment in the legacy SSA numbering (measured: 8 renders
# mid-screen, 6 at the top): 2 = bottom centre, 6 = top centre.
SUBTITLE_POSITIONS = {"bottom": 2, "top": 6}
SUBTITLE_STYLE_FIELDS = (
    "font", "font_size", "text_colour", "outline_colour", "outline", "position", "margin",
)
SUBTITLE_COLOUR_PATTERN = "^#[0-9A-Fa-f]{6}$"
