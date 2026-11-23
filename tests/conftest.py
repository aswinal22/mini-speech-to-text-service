from pathlib import Path
import pytest
from fastapi.testclient import TestClient

import app
from app.main import app as fastapi_app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session-scoped TestClient fixture."""
    return TestClient(fastapi_app)


@pytest.fixture(scope="session")
def english_wav_path() -> Path:
    """Return path to valid English WAV sample."""
    path = Path(__file__).resolve().parent.parent / "samples" / "english_sample.wav"
    assert path.exists(), f"Sample audio not found at {path}"
    return path


@pytest.fixture(scope="session")
def hindi_wav_path() -> Path:
    """Return path to valid Hindi WAV sample."""
    path = Path(__file__).resolve().parent.parent / "samples" / "hindi_sample.wav"
    assert path.exists(), f"Sample audio not found at {path}"
    return path
