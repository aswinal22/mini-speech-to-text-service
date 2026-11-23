from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class TranscriptionSummaryResponse(BaseModel):
    """Abbreviated summary representation of a transcription for list responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Unique record identifier")
    filename: str = Field(..., description="Original name of the audio file")
    detected_language: str = Field(..., description="ISO language code detected by Whisper")
    transcript: str = Field(..., description="Output transcript produced by the ASR model")
    wer: Optional[float] = Field(
        None, description="Word Error Rate compared to reference transcript, if provided"
    )
    created_at: str = Field(..., description="Timestamp of transcription creation")


class TranscriptionDetailResponse(BaseModel):
    """Comprehensive detail representation of a transcription record."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Unique record identifier")
    filename: str = Field(..., description="Original name of the audio file")
    detected_language: str = Field(..., description="ISO language code detected by Whisper")
    language_probability: float = Field(
        ..., description="Model confidence score for the detected language"
    )
    transcript: str = Field(..., description="Output transcript produced by the ASR model")
    reference_transcript: Optional[str] = Field(
        None, description="Ground-truth reference transcript if supplied"
    )
    wer: Optional[float] = Field(
        None, description="Calculated Word Error Rate (WER) against reference transcript"
    )
    duration_seconds: float = Field(..., description="Audio duration in seconds")
    inference_time_seconds: float = Field(
        ..., description="Execution time taken by Whisper model inference in seconds"
    )
    created_at: str = Field(..., description="Timestamp of transcription creation")


class SearchResultResponse(BaseModel):
    """Search match entry with contextual text snippet."""

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Record identifier")
    filename: str = Field(..., description="Filename of original recording")
    detected_language: str = Field(..., description="Detected language code")
    transcript: str = Field(..., description="Full transcript text")
    match_snippet: str = Field(..., description="Contextual excerpt containing the query")
    created_at: str = Field(..., description="Creation timestamp")


class ErrorResponse(BaseModel):
    """Standardized error payload for API error responses."""

    detail: str = Field(..., description="Human-readable error description")
    error_type: str = Field(..., description="Error classification category")
