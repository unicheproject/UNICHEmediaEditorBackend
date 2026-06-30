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
    # Comma-separated allowed origins, or "*" to allow any (dev default).
    cors_allow_origins: str = "*"

    @property
    def cors_origins(self) -> list[str]:
        raw = self.cors_allow_origins.strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    # --- Auth (Keycloak resource server) + Catalogue integration ---
    # Issuer string MUST equal what the browser sees and what the catalogue
    # validates against, e.g. https://idp.uniche-eccch.eu/realms/uniche.
    idp_issuer_uri: str = ""
    required_audience: str = "uniche-platform"
    # JWKS are cached for this long (seconds) before refetch.
    auth_jwks_cache_seconds: int = 3600
    # Base URL of the catalogue (authorization authority), no trailing /api/v1.
    catalogue_base_url: str = ""
    catalogue_timeout_seconds: float = 10.0
    # This tool's slug in the catalogue's authoring_tools registry.
    tool_slug: str = "media-editor"

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

    # --- Agent planner ---
    agent_provider: str = "mock"  # "mock" | "openrouter"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "qwen/qwen3.6-flash"
    agent_timeout_seconds: float = 120.0
    agent_max_repair_retries: int = 2
    # Restrict the agent to deterministic (local-tool) capabilities only,
    # excluding hosted-AI / GPU ops from the planner catalog and validation.
    agent_deterministic_only: bool = True

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    # Accepted upload extensions per media type (lowercase, no leading dot)
    allowed_image_extensions: set[str] = {"jpg", "jpeg", "png", "webp", "tiff"}
    allowed_audio_extensions: set[str] = {"mp3", "wav", "m4a", "flac", "ogg"}
    allowed_video_extensions: set[str] = {"mp4", "mov", "webm", "mkv"}
    allowed_subtitle_extensions: set[str] = {"srt", "vtt"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
