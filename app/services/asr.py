import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Union

import librosa
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from app.config import get_settings


@dataclass
class ASRResult:
    """Structured output representation from Whisper speech recognition."""

    transcript: str
    detected_language: str
    language_probability: float
    duration_seconds: float
    inference_time_seconds: float


class AudioProcessingError(Exception):
    """Raised when audio decoding or model inference encounters an unrecoverable error."""

    pass


def load_audio_waveform(audio_source: Union[str, Path, BinaryIO]) -> np.ndarray:
    """
    Read an audio file or stream using soundfile and convert it to a 16kHz mono float32 waveform.
    Bypasses PyAV version quirks and external ffmpeg dependencies.
    """
    try:
        source = str(audio_source) if isinstance(audio_source, Path) else audio_source
        audio_data, sample_rate = sf.read(source, dtype="float32")

        # Convert multi-channel / stereo to mono by averaging channels
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample to 16 kHz if necessary
        if sample_rate != 16000:
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

        if len(audio_data) == 0:
            raise AudioProcessingError("Audio waveform is empty.")

        return audio_data
    except Exception as e:
        if isinstance(e, AudioProcessingError):
            raise
        raise AudioProcessingError(f"Failed to decode audio waveform: {e}") from e


class WhisperASRService:
    """
    Thread-safe Whisper ASR Service wrapping faster-whisper.
    Configured specifically for CPU INT8 execution and controlled CPU threading.
    """

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
        cpu_threads: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.model_size = model_size or settings.whisper_model_size
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self.cpu_threads = cpu_threads or settings.whisper_cpu_threads
        self._model: Optional[WhisperModel] = None
        self._lock = threading.Lock()

    def get_model(self) -> WhisperModel:
        """Lazy-load the WhisperModel singleton with thread safety."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        self._model = WhisperModel(
                            model_size_or_path=self.model_size,
                            device=self.device,
                            compute_type=self.compute_type,
                            cpu_threads=self.cpu_threads,
                        )
                    except Exception as e:
                        raise AudioProcessingError(
                            f"Failed to load Whisper model ({self.model_size}): {e}"
                        ) from e
        return self._model

    def transcribe(
        self,
        audio_source: Union[str, Path, BinaryIO],
        language: Optional[str] = None,
    ) -> ASRResult:
        """
        Transcribe an audio file path or stream.
        - Measures exact inference time in seconds.
        - Emits language detection confidence.
        - Concatenates segment text.
        - Raises AudioProcessingError on corrupt or unreadable audio.
        """
        # Load audio data as 16kHz mono numpy array
        waveform = load_audio_waveform(audio_source)
        audio_duration = len(waveform) / 16000.0

        model = self.get_model()
        start_time = time.perf_counter()

        try:
            segments, info = model.transcribe(
                waveform,
                language=language,
                beam_size=5,
                vad_filter=True,
            )

            # Consume the segments generator to complete transcription
            segment_texts = [segment.text.strip() for segment in segments]
            transcript = " ".join(segment_texts).strip()

            elapsed_time = time.perf_counter() - start_time
            duration = float(info.duration) if hasattr(info, "duration") and info.duration else audio_duration

            return ASRResult(
                transcript=transcript,
                detected_language=info.language or "unknown",
                language_probability=round(float(info.language_probability), 4)
                if hasattr(info, "language_probability") and info.language_probability
                else 1.0,
                duration_seconds=round(duration, 2),
                inference_time_seconds=round(elapsed_time, 4),
            )
        except Exception as e:
            if isinstance(e, AudioProcessingError):
                raise
            raise AudioProcessingError(f"Audio transcription failed: {e}") from e


_asr_instance: Optional[WhisperASRService] = None
_instance_lock = threading.Lock()


def get_asr_service() -> WhisperASRService:
    """Dependency provider returning singleton WhisperASRService instance."""
    global _asr_instance
    if _asr_instance is None:
        with _instance_lock:
            if _asr_instance is None:
                _asr_instance = WhisperASRService()
    return _asr_instance
