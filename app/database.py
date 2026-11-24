import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from app.config import get_settings


def get_db_path(custom_path: Path | None = None) -> Path:
    """Resolve database path and ensure parent directories exist."""
    db_path = custom_path or get_settings().db_path
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


@contextmanager
def get_db_connection(
    custom_path: Path | None = None,
) -> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding an SQLite connection configured with row_factory=sqlite3.Row."""
    db_path = get_db_path(custom_path)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(custom_path: Path | None = None) -> None:
    """Initialize SQLite database tables, regional metric columns, and indexes."""
    with get_db_connection(custom_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                detected_language TEXT NOT NULL,
                language_probability REAL NOT NULL DEFAULT 1.0,
                transcript TEXT NOT NULL,
                reference_transcript TEXT,
                wer REAL,
                cer REAL,
                aer REAL,
                acoustic_accuracy REAL,
                duration_seconds REAL NOT NULL DEFAULT 0.0,
                inference_time_seconds REAL NOT NULL DEFAULT 0.0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Migrate existing tables dynamically if columns are absent
        cursor.execute("PRAGMA table_info(transcriptions)")
        existing_cols = {row["name"] for row in cursor.fetchall()}
        for col, col_type in [
            ("cer", "REAL"),
            ("aer", "REAL"),
            ("acoustic_accuracy", "REAL"),
        ]:
            if col not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE transcriptions ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_transcriptions_created_at
            ON transcriptions(created_at DESC)
            """
        )


def save_transcription(
    filename: str,
    detected_language: str,
    language_probability: float,
    transcript: str,
    reference_transcript: str | None,
    wer: float | None,
    duration_seconds: float,
    inference_time_seconds: float,
    cer: float | None = None,
    aer: float | None = None,
    acoustic_accuracy: float | None = None,
    custom_path: Path | None = None,
) -> int:
    """Insert a new transcription record into the database and return the row ID."""
    with get_db_connection(custom_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO transcriptions (
                filename,
                detected_language,
                language_probability,
                transcript,
                reference_transcript,
                wer,
                cer,
                aer,
                acoustic_accuracy,
                duration_seconds,
                inference_time_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                filename,
                detected_language,
                float(language_probability),
                transcript,
                reference_transcript,
                float(wer) if wer is not None else None,
                float(cer) if cer is not None else None,
                float(aer) if aer is not None else None,
                float(acoustic_accuracy) if acoustic_accuracy is not None else None,
                float(duration_seconds),
                float(inference_time_seconds),
            ),
        )
        record_id = cursor.lastrowid
        if record_id is None:
            raise RuntimeError("Failed to retrieve lastrowid after inserting transcription record")
        return record_id


def get_all_transcriptions(
    limit: int = 50,
    offset: int = 0,
    custom_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Retrieve paginated transcriptions ordered from newest to oldest."""
    with get_db_connection(custom_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, detected_language, transcript, wer, cer, created_at
            FROM transcriptions
            ORDER BY created_at DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = cursor.fetchall()
        return [dict(row) for row in rows]


def get_transcription_by_id(
    record_id: int,
    custom_path: Path | None = None,
) -> dict[str, Any] | None:
    """Retrieve a single transcription record by ID with all regional error metrics."""
    with get_db_connection(custom_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, detected_language, language_probability,
                   transcript, reference_transcript, wer, cer, aer, acoustic_accuracy,
                   duration_seconds, inference_time_seconds, created_at
            FROM transcriptions
            WHERE id = ?
            """,
            (record_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def search_transcripts(
    query: str,
    limit: int = 20,
    custom_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Search transcripts using case-insensitive substring matching and return snippets."""
    clean_query = query.strip()
    if not clean_query:
        return []

    with get_db_connection(custom_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, filename, detected_language, transcript, created_at
            FROM transcriptions
            WHERE transcript LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (f"%{clean_query}%", limit),
        )
        rows = cursor.fetchall()
        results: list[dict[str, Any]] = []
        for row in rows:
            data = dict(row)
            text = data["transcript"]
            pos = text.lower().find(clean_query.lower())
            if pos != -1:
                start = max(0, pos - 20)
                end = min(len(text), pos + len(clean_query) + 20)
                snippet = f"...{text[start:end]}..."
            else:
                snippet = text[:50]
            data["match_snippet"] = snippet
            results.append(data)
        return results
