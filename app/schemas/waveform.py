"""Bounded waveform preview contract, independent of technical metadata."""

from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.enums import WaveformReason

WAVEFORM_VERSION: Final[Literal[1]] = 1
MAX_WAVEFORM_BUCKETS = 1024
WAVEFORM_SAMPLE_RATE: Final[Literal[8000]] = 8000

Amplitude = Annotated[float, Field(ge=-1, le=1, allow_inf_nan=False)]


class WaveformAvailable(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["available"] = "available"
    version: Literal[1] = WAVEFORM_VERSION
    channels: Literal[1] = 1
    sample_rate: Literal[8000] = WAVEFORM_SAMPLE_RATE
    bucket_samples: int = Field(ge=1)
    bucket_count: int = Field(ge=1, le=MAX_WAVEFORM_BUCKETS)
    duration_seconds: float = Field(gt=0, allow_inf_nan=False)
    peaks: list[tuple[Amplitude, Amplitude]] = Field(
        min_length=1, max_length=MAX_WAVEFORM_BUCKETS,
        description="Chronological min/max pairs in [-1,1] after first-stream mono downmix. "
        "Each bucket spans bucket_samples/sample_rate seconds; only the last may be shorter.",
    )

    @model_validator(mode="after")
    def consistent_envelope(self) -> "WaveformAvailable":
        if len(self.peaks) != self.bucket_count or any(lo > hi for lo, hi in self.peaks):
            raise ValueError("Invalid waveform envelope")
        start = (self.bucket_count - 1) * self.bucket_samples / self.sample_rate
        end = self.bucket_count * self.bucket_samples / self.sample_rate
        if not start < self.duration_seconds <= end:
            raise ValueError("Invalid waveform duration")
        return self


class WaveformState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["pending", "unavailable"]
    reason_code: WaveformReason | None = None


WaveformResponse = Annotated[
    WaveformAvailable | WaveformState, Field(discriminator="status")
]
