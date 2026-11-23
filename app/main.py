import asyncio
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Path as PathParam, Query, UploadFile, status
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.database import (
    get_all_transcriptions,
    get_transcription_by_id,
    init_db,
    save_transcription,
    search_transcripts,
)
from app.schemas import (
    ErrorResponse,
    SearchResultResponse,
    TranscriptionDetailResponse,
    TranscriptionSummaryResponse,
)
from app.services.asr import AudioProcessingError, get_asr_service
from app.services.evaluator import calculate_wer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize database on startup."""
    init_db()
    yield


app = FastAPI(
    title="PlanetRead Mini Speech-to-Text Service",
    description="ASR service with Whisper CPU inference, SQLite storage, and Word Error Rate (WER) evaluation.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(AudioProcessingError)
async def audio_processing_exception_handler(request, exc: AudioProcessingError):
    """Handle audio decoding and model inference exceptions."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"detail": str(exc), "error_type": "AudioProcessingError"},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle standard HTTP exceptions with structured error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_type": "HTTPException"},
    )


@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check():
    """Verify service health and configured model parameters."""
    settings = get_settings()
    return {
        "status": "healthy",
        "app_name": settings.app_name,
        "version": settings.version,
        "model_size": settings.whisper_model_size,
        "compute_type": settings.whisper_compute_type,
        "device": settings.whisper_device,
    }


@app.post(
    "/transcribe",
    response_model=TranscriptionDetailResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid audio file or empty payload"},
        500: {"model": ErrorResponse, "description": "Internal server processing error"},
    },
    tags=["ASR"],
    summary="Transcribe audio file (.wav or .mp3) and compute WER if reference provided",
)
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (.wav or .mp3)"),
    reference: Optional[str] = Form(None, description="Optional ground-truth reference transcript"),
) -> TranscriptionDetailResponse:
    """
    Accepts an audio file (.wav or .mp3) and an optional reference transcript.
    - Validates file extension and non-empty content.
    - Executes speech recognition on CPU via faster-whisper in a threadpool.
    - Computes Word Error Rate (WER) if reference is provided.
    - Persists results to SQLite.
    - Returns full transcription details.
    """
    settings = get_settings()
    filename = file.filename or "unknown_audio"
    file_ext = Path(filename).suffix.lower()

    # Validate file extension
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{file_ext}'. Allowed types: {', '.join(sorted(settings.allowed_extensions))}",
        )

    # Read audio content safely
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded audio file: {e}",
        ) from e

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty (0 bytes).",
        )

    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds maximum permitted limit ({settings.max_upload_size_bytes // (1024*1024)} MB).",
        )

    # Write content to temporary file for Whisper processing
    temp_dir = tempfile.gettempdir()
    temp_file = Path(temp_dir) / f"upload_{os.urandom(8).hex()}{file_ext}"

    try:
        temp_file.write_bytes(content)

        # Offload CPU-bound inference to worker thread
        asr_service = get_asr_service()
        asr_result = await asyncio.to_thread(asr_service.transcribe, temp_file)

        # Calculate Word Error Rate (WER) if reference was supplied
        wer_value = calculate_wer(
            hypothesis=asr_result.transcript,
            reference=reference,
        )

        # Save record to SQLite
        record_id = save_transcription(
            filename=filename,
            detected_language=asr_result.detected_language,
            language_probability=asr_result.language_probability,
            transcript=asr_result.transcript,
            reference_transcript=reference.strip() if reference else None,
            wer=wer_value,
            duration_seconds=asr_result.duration_seconds,
            inference_time_seconds=asr_result.inference_time_seconds,
        )

        saved_record = get_transcription_by_id(record_id)
        if not saved_record:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve persisted transcription record.",
            )

        return TranscriptionDetailResponse.model_validate(saved_record)

    finally:
        # Guarantee cleanup of temporary file
        if temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass


@app.get(
    "/transcripts",
    response_model=List[TranscriptionSummaryResponse],
    tags=["Transcripts"],
    summary="List past transcriptions",
)
async def list_transcriptions(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of records to return"),
    offset: int = Query(0, ge=0, description="Offset index for pagination"),
) -> List[TranscriptionSummaryResponse]:
    """Retrieve a list of past transcriptions ordered newest first."""
    records = get_all_transcriptions(limit=limit, offset=offset)
    return [TranscriptionSummaryResponse.model_validate(record) for record in records]


@app.get(
    "/transcripts/{id}",
    response_model=TranscriptionDetailResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Transcription not found"},
    },
    tags=["Transcripts"],
    summary="Get transcription details by ID",
)
async def get_transcription(
    id: int = PathParam(..., ge=1, description="Unique transcription record identifier"),
) -> TranscriptionDetailResponse:
    """Retrieve full details of a specific transcription by its ID."""
    record = get_transcription_by_id(id)
    if not record:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription with id {id} not found.",
        )
    return TranscriptionDetailResponse.model_validate(record)


@app.get(
    "/search",
    response_model=List[SearchResultResponse],
    tags=["Search"],
    summary="Search stored transcripts by query string",
)
async def search_stored_transcripts(
    q: str = Query(..., min_length=1, description="Keyword or phrase to search within transcripts"),
    limit: int = Query(20, ge=1, le=50, description="Maximum search results to return"),
) -> List[SearchResultResponse]:
    """Search stored transcripts for matching terms and return excerpt snippets."""
    records = search_transcripts(query=q, limit=limit)
    return [SearchResultResponse.model_validate(record) for record in records]
