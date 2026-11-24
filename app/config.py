import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration with hardware-aware defaults for local CPU execution."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Mini Speech-to-Text Service"
    version: str = "1.0.0"
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    db_path: Path = data_dir / "transcriptions.db"
    samples_dir: Path = base_dir / "samples"
    models_dir: Path = base_dir / "models" / "small"

    # ASR Hardware Optimization Settings (Whisper small INT8, ~450MB RAM)
    whisper_model_size: str = (
        str(Path(__file__).resolve().parent.parent / "models" / "small")
        if (Path(__file__).resolve().parent.parent / "models" / "small").exists()
        else "small"
    )
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_cpu_threads: int = min(8, os.cpu_count() or 6)

    # Audio Constraints (supports standard formats including mobile recordings)
    allowed_extensions: set[str] = {".wav", ".mp3", ".m4a", ".ogg"}
    max_upload_size_bytes: int = 25 * 1024 * 1024  # 25 MB


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
