"""Application settings, loaded from environment variables / .env."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- App ---
    app_name: str = "UNICHE Media Editor API"
    api_v1_prefix: str = "/api/v1"
    log_level: str = "INFO"

    # --- Database ---
    database_url: str = "postgresql+asyncpg://uniche:uniche@localhost:5432/uniche"

    # --- Redis / job queue ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Storage ---
    storage_dir: str = "./storage"
    max_upload_size_mb: int = 200

    # --- Inference provider ---
    inference_provider: str = "mock"  # "mock" | "http"
    inference_base_url: str = ""
    inference_api_key: str = ""
    inference_timeout_seconds: float = 120.0
    inference_image_caption_path: str = "/image/caption"
    inference_audio_transcribe_path: str = "/audio/transcribe"

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    # Accepted upload extensions per media type (lowercase, no leading dot)
    allowed_image_extensions: set[str] = {"jpg", "jpeg", "png", "webp", "tiff"}
    allowed_audio_extensions: set[str] = {"mp3", "wav", "m4a", "flac", "ogg"}
    allowed_video_extensions: set[str] = {"mp4", "mov", "webm", "mkv"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
