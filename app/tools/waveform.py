"""Streaming first-audio-stream waveform reduction; no duration-sized buffers."""

import asyncio
import sys
from array import array
from pathlib import Path

from app.schemas.waveform import (
    MAX_WAVEFORM_BUCKETS,
    WAVEFORM_SAMPLE_RATE,
    WaveformAvailable,
)
from app.tools.process_policy import media_process

WAVEFORM_TIMEOUT_SECONDS = 120.0
PCM_CHUNK_BYTES = 64 * 1024


class WaveformGenerationError(Exception):
    """Non-diagnostic preview failure; never exposes FFmpeg stderr or paths."""


class EnvelopeAccumulator:
    """Merge adjacent full buckets as needed, keeping the last partial bucket separate."""

    def __init__(self) -> None:
        self.bucket_samples = 256
        self.peaks: list[tuple[int, int]] = []
        self.total_samples = 0
        self.filled = 0
        self.minimum = 32767
        self.maximum = -32768

    def add(self, samples: array) -> None:
        self.total_samples += len(samples)
        offset = 0
        while offset < len(samples):
            if len(self.peaks) == MAX_WAVEFORM_BUCKETS:
                self.peaks = [
                    (min(a[0], b[0]), max(a[1], b[1]))
                    for a, b in zip(self.peaks[::2], self.peaks[1::2], strict=True)
                ]
                self.bucket_samples *= 2
            take = min(self.bucket_samples - self.filled, len(samples) - offset)
            chunk = samples[offset:offset + take]
            self.minimum = min(self.minimum, min(chunk))
            self.maximum = max(self.maximum, max(chunk))
            self.filled += take
            offset += take
            if self.filled == self.bucket_samples:
                self.peaks.append((self.minimum, self.maximum))
                self.filled = 0
                self.minimum, self.maximum = 32767, -32768

    def finish(self) -> WaveformAvailable:
        if not self.total_samples:
            raise WaveformGenerationError("No decoded audio")
        peaks = self.peaks + ([(self.minimum, self.maximum)] if self.filled else [])
        return WaveformAvailable(
            bucket_samples=self.bucket_samples,
            bucket_count=len(peaks),
            duration_seconds=self.total_samples / WAVEFORM_SAMPLE_RATE,
            peaks=[(lo / 32768, hi / 32768) for lo, hi in peaks],
        )


async def generate_waveform(path: Path) -> WaveformAvailable:
    """Decode fixed-rate mono signed PCM incrementally, killing the child on all exits."""
    args = [
        "ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-filter_threads", "1",
        "-i", str(path.absolute()), "-map", "0:a:0", "-vn", "-ac", "1", "-ar",
        str(WAVEFORM_SAMPLE_RATE), "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
    ]
    accumulator = EnvelopeAccumulator()
    remainder = b""
    try:
        async with media_process(
            args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
            limit=PCM_CHUNK_BYTES,
        ) as proc:
            assert proc.stdout is not None
            async with asyncio.timeout(WAVEFORM_TIMEOUT_SECONDS):
                while chunk := await proc.stdout.read(PCM_CHUNK_BYTES):
                    chunk = remainder + chunk
                    usable = len(chunk) - len(chunk) % 2
                    remainder = chunk[usable:]
                    samples = array("h")
                    samples.frombytes(chunk[:usable])
                    if sys.byteorder != "little":
                        samples.byteswap()
                    accumulator.add(samples)
                if await proc.wait() != 0 or remainder:
                    raise WaveformGenerationError("Audio decoding failed")
            return accumulator.finish()
    except TimeoutError as exc:
        raise WaveformGenerationError("Decoder timeout") from exc
    except OSError as exc:
        raise WaveformGenerationError("Decoder unavailable") from exc
