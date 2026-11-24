from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.services.evaluator import (
    calculate_cer,
    calculate_regional_error_metrics,
    calculate_wer,
    normalize_text,
    tokenize_indic_aksharas,
)


def test_health_check(client: TestClient) -> None:
    """Verify that the health check endpoint returns 200 and valid system details."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_size" in data
    assert data["device"] == "cpu"
    assert data["compute_type"] == "int8"


def test_transcribe_wav_with_reference(client: TestClient, english_wav_path: Path) -> None:
    """Verify POST /transcribe accepts WAV, runs ASR, calculates WER, and persists result."""
    reference_text = (
        "And so my fellow Americans, ask not what your country can do for you, "
        "ask what you can do for your country."
    )
    with open(english_wav_path, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("english_sample.wav", f, "audio/wav")},
            data={"reference": reference_text},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["id"] > 0
    assert data["filename"] == "english_sample.wav"
    assert data["detected_language"] == "en"
    assert data["language_probability"] > 0.5
    assert len(data["transcript"]) > 0
    assert data["reference_transcript"] == reference_text
    assert data["wer"] == 0.0  # Exact match for JFK sample
    assert data["cer"] == 0.0
    assert data["aer"] == 0.0
    assert data["acoustic_accuracy"] == 1.0
    assert data["duration_seconds"] > 0
    assert data["inference_time_seconds"] > 0


def test_transcribe_without_reference(client: TestClient, english_wav_path: Path) -> None:
    """Verify POST /transcribe succeeds without reference transcript and sets wer=None."""
    with open(english_wav_path, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("english_sample.wav", f, "audio/wav")},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["id"] > 0
    assert data["reference_transcript"] is None
    assert data["wer"] is None


def test_transcribe_hindi_sample(client: TestClient, hindi_wav_path: Path) -> None:
    """Verify POST /transcribe correctly identifies Hindi speech."""
    with open(hindi_wav_path, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("hindi_sample.wav", f, "audio/wav")},
        )

    assert response.status_code == 201
    data = response.json()
    assert data["id"] > 0
    assert data["detected_language"] == "hi"
    assert len(data["transcript"]) > 0


def test_transcribe_invalid_extension(client: TestClient) -> None:
    """Verify POST /transcribe rejects unsupported file extensions with 400 Bad Request."""
    response = client.post(
        "/transcribe",
        files={"file": ("document.txt", b"plain text data", "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported file type" in data["detail"]
    assert data["error_type"] == "HTTPException"


def test_transcribe_empty_file(client: TestClient) -> None:
    """Verify POST /transcribe rejects 0-byte audio files with 400 Bad Request."""
    response = client.post(
        "/transcribe",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()


def test_list_transcriptions(client: TestClient) -> None:
    """Verify GET /transcripts returns paginated summary list."""
    response = client.get("/transcripts?limit=10&offset=0")
    assert response.status_code == 200
    records = response.json()
    assert isinstance(records, list)
    assert len(records) > 0
    record = records[0]
    assert "id" in record
    assert "filename" in record
    assert "detected_language" in record
    assert "transcript" in record
    assert "created_at" in record


def test_get_transcription_by_id_found(client: TestClient) -> None:
    """Verify GET /transcripts/{id} returns single record details."""
    # Fetch latest ID from listing
    list_res = client.get("/transcripts?limit=1")
    record_id = list_res.json()[0]["id"]

    response = client.get(f"/transcripts/{record_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == record_id
    assert "duration_seconds" in data
    assert "inference_time_seconds" in data


def test_get_transcription_by_id_not_found(client: TestClient) -> None:
    """Verify GET /transcripts/{id} returns 404 when ID does not exist."""
    response = client.get("/transcripts/9999999")
    assert response.status_code == 404
    data = response.json()
    assert "not found" in data["detail"].lower()


def test_search_transcripts_endpoint(client: TestClient) -> None:
    """Verify GET /search returns matches with context snippet."""
    response = client.get("/search?q=country")
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) > 0
    assert "match_snippet" in results[0]
    assert "country" in results[0]["match_snippet"].lower()


def test_wer_evaluator_unit_cases() -> None:
    """Comprehensive unit testing of text normalization and WER calculation."""
    # English normalization
    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text("  Testing... 1, 2, 3!  ") == "testing 1 2 3"

    # Hindi / Devanagari normalization preserving matras and viramas
    # Automatically handles dandas (। and ॥) via Unicode category 'P*' without hardcoding
    assert normalize_text("नमस्ते, दुनिया।") == "नमस्ते दुनिया"
    assert normalize_text("भारत देश॥") == "भारत देश"

    # Invisible formatting characters (ZWNJ \u200c and ZWJ \u200d in Indic scripts)
    assert normalize_text("नमस्\u200cते") == "नमस्ते"
    assert normalize_text("தமி\u200dழ்") == "தமிழ்"
    assert calculate_wer("नमस्ते", "नमस्\u200cते") == 0.0
    assert calculate_wer("தமிழ்", "தமி\u200dழ்") == 0.0

    # International punctuation (Arabic comma ،, question mark ؟)
    assert normalize_text("مرحبا، بالعالم؟") == "مرحبا بالعالم"

    # WER exact match
    assert calculate_wer("Hello world", "Hello, World!") == 0.0
    assert calculate_wer("नमस्ते दुनिया", "नमस्ते, दुनिया।") == 0.0

    # WER substitution & insertion
    # ref: "one two three" (3 words), hyp: "one four three five" (sub 1, ins 1 -> WER = 2/3 = 0.6667)
    wer = calculate_wer("one four three five", "one two three")
    assert wer == pytest.approx(0.6667, abs=1e-3)

    # Empty cases
    assert calculate_wer("anything", None) is None
    assert calculate_wer("", "") == 0.0
    assert calculate_wer("", "reference words") == 1.0


def test_regional_error_metrics_calculation() -> None:
    """Verify Character Error Rate (CER), Akshara Error Rate (AER), and acoustic accuracy."""
    # Tamil Sandhi split test: "வணக்கம் நண்பர்களே" vs "வணக்கம் நண்பர் களே"
    ref_ta = "வணக்கம் நண்பர்களே"
    hyp_ta = "வணக்கம் நண்பர் களே"

    # In WER: 1 sub + 1 ins -> WER is 0.5 or higher
    wer = calculate_wer(hyp_ta, ref_ta)
    assert wer >= 0.5

    # In CER: only 1 space difference -> CER is low (< 0.10)
    cer = calculate_cer(hyp_ta, ref_ta)
    assert cer is not None
    assert cer < 0.10

    # In AER: Syllable clusters are segmented accurately
    aksharas = tokenize_indic_aksharas("வணக்கம்")
    assert len(aksharas) == 5  # வ, ண, க், க, ம் -> 5 syllabic units

    # Full regional metric dictionary
    metrics = calculate_regional_error_metrics(hyp_ta, ref_ta, language="ta")
    assert metrics["wer"] is not None
    assert metrics["cer"] is not None
    assert metrics["aer"] is not None
    assert metrics["acoustic_accuracy"] is not None
    assert metrics["acoustic_accuracy"] > 0.90
