import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

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


def load_audio_waveform(audio_source: str | Path | BinaryIO) -> np.ndarray:
    """
    Read an audio file or stream and convert it to a normalized 16kHz mono float32 waveform.
    Uses soundfile first, with automatic fallback to PyAV for formats like .m4a and .ogg.
    Applies peak loudness normalization to maximize acoustic feature resolution.
    """
    try:
        source = str(audio_source) if isinstance(audio_source, Path) else audio_source
        audio_data = None
        sample_rate = 16000

        # Attempt decoding with soundfile
        try:
            audio_data, sample_rate = sf.read(source, dtype="float32")
            if audio_data.ndim > 1:
                audio_data = np.mean(audio_data, axis=1)
            if sample_rate != 16000:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        except Exception:  # noqa: BLE001
            # Fallback to PyAV for containers like M4A / AAC / OGG
            import av

            container = av.open(source)
            resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
            frames = []
            for frame in container.decode(audio=0):
                for r_frame in resampler.resample(frame):
                    frames.append(r_frame.to_ndarray())
            if not frames:
                raise AudioProcessingError("No audio frames decoded from file.") from None
            audio_data = np.concatenate(frames, axis=1)[0].astype(np.float32) / 32768.0

        if audio_data is None or len(audio_data) == 0:
            raise AudioProcessingError("Decoded audio waveform is empty.")

        # Peak loudness normalization (scales speech cleanly to 0.95 peak, recovering quiet audio)
        peak = float(np.max(np.abs(audio_data)))
        if peak > 1e-6:
            audio_data = (audio_data / peak) * 0.95

        return audio_data.astype(np.float32)
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
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        cpu_threads: int | None = None,
    ) -> None:
        settings = get_settings()
        self.model_size = model_size or settings.whisper_model_size
        self.device = device or settings.whisper_device
        self.compute_type = compute_type or settings.whisper_compute_type
        self.cpu_threads = cpu_threads or settings.whisper_cpu_threads
        self._model: WhisperModel | None = None
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
        audio_source: str | Path | BinaryIO,
        language: str | None = None,
        prompt: str | None = None,
        beam_size: int = 1,
    ) -> ASRResult:
        """
        Transcribe an audio file path or stream.
        - language: Optional ISO language code (e.g., 'en', 'ta', 'hi', 'ml').
                    When supplied, bypasses the 99-language detection pass for ~30% faster inference.
        - prompt: Optional initial prompt string to bias vocabulary/spelling and force native script.
        - beam_size: 1 for fast greedy CPU decoding (default), 5 for exhaustive search.
        """
        waveform = load_audio_waveform(audio_source)
        audio_duration = len(waveform) / 16000.0

        model = self.get_model()
        start_time = time.perf_counter()

        try:
            segments, info = model.transcribe(
                waveform,
                language=language if language and language.strip() else None,
                initial_prompt=prompt if prompt and prompt.strip() else None,
                beam_size=beam_size,
                best_of=max(1, beam_size),
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )

            # Consume the segments generator to complete transcription
            segment_texts = [segment.text.strip() for segment in segments]
            transcript = " ".join(segment_texts).strip()

            elapsed_time = time.perf_counter() - start_time
            duration = (
                float(info.duration)
                if hasattr(info, "duration") and info.duration
                else audio_duration
            )

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


_asr_instance: WhisperASRService | None = None
_instance_lock = threading.Lock()


def get_asr_service() -> WhisperASRService:
    """Dependency provider returning singleton WhisperASRService instance."""
    global _asr_instance
    if _asr_instance is None:
        with _instance_lock:
            if _asr_instance is None:
                _asr_instance = WhisperASRService()
    return _asr_instance
