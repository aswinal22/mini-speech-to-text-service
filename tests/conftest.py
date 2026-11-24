from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.database import init_db
from app.main import app as fastapi_app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session-scoped TestClient fixture."""
    init_db()
    return TestClient(fastapi_app)


@pytest.fixture(scope="session")
def english_wav_path() -> Path:
    """Return path to valid English audio sample."""
    samples_dir = Path(__file__).resolve().parent.parent / "samples"
    primary = samples_dir / "english_1_jfk.wav"
    if primary.exists():
        return primary
    fallback = samples_dir / "english_sample.wav"
    if fallback.exists():
        return fallback
    any_en = list(samples_dir.glob("english*.wav"))
    assert any_en, f"No English audio sample found in {samples_dir}"
    return any_en[0]


@pytest.fixture(scope="session")
def hindi_wav_path() -> Path:
    """Return path to valid Hindi audio sample."""
    samples_dir = Path(__file__).resolve().parent.parent / "samples"
    primary = samples_dir / "tts_1795456188137.mp3"
    if primary.exists():
        return primary
    fallback = samples_dir / "hindi_sample.wav"
    if fallback.exists():
        return fallback
    any_hi = list(samples_dir.glob("tts*.mp3"))
    assert any_hi, f"No Hindi audio sample found in {samples_dir}"
    return any_hi[0]
