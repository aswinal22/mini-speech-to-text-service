from functools import lru_cache
from pathlib import Path
from typing import Set
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
    models_dir: Path = base_dir / "models" / "base"

    # ASR Hardware Optimization Settings (Intel 8-thread CPU, ~1.1GB RAM)
    # Uses locally cached workspace model directory if present, else model identifier
    whisper_model_size: str = (
        str(Path(__file__).resolve().parent.parent / "models" / "base")
        if (Path(__file__).resolve().parent.parent / "models" / "base").exists()
        else "base"
    )
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"  # 8-bit quantization for minimal memory usage
    whisper_cpu_threads: int = 4  # Reserve 4 logical cores for system & API responsiveness

    # Audio Constraints
    allowed_extensions: Set[str] = {".wav", ".mp3"}
    max_upload_size_bytes: int = 25 * 1024 * 1024  # 25 MB


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings instance."""
    return Settings()
