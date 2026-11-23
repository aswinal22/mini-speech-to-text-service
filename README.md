# Mini Speech-to-Text Service (PlanetRead Build Assignment)

A lightweight, production-ready Automatic Speech Recognition (ASR) backend service built with FastAPI, Whisper (via `faster-whisper`), SQLite, and `jiwer` for Word Error Rate (WER) evaluation. Tailored for local CPU inference and speech recognition in Indian languages.

---

## 🚀 How to Run (In 3 Commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the server
uvicorn app.main:app --host 127.0.0.1 --port 8000

# 3. Transcribe a sample audio clip with reference text
curl -X POST "http://127.0.0.1:8000/transcribe" -F "file=@samples/english_sample.wav" -F "reference=And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
```

Interactive Swagger API documentation is available at: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

To run the automated test suite:
```bash
pytest tests/ -v
```

---

## 🛠️ Architecture & Design

```
mini-speech-to-text-service/
├── app/
│   ├── __init__.py          # Windows DLL initialization for C-extensions
│   ├── config.py            # Hardware-aware configuration (Pydantic Settings)
│   ├── database.py          # SQLite persistence, thread-safe connection pool & search
│   ├── main.py              # FastAPI endpoints, validation & error handlers
│   ├── schemas.py           # Pydantic request/response schemas
│   └── services/
│       ├── __init__.py
│       ├── asr.py           # Thread-safe Whisper ASR service (INT8 CPU)
│       └── evaluator.py     # Multilingual text normalization & WER computation
├── models/base/             # Offline cached faster-whisper base model weights
├── samples/
│   ├── english_sample.wav   # 11.0s JFK speech sample (16kHz mono)
│   └── hindi_sample.wav     # 12.0s Hindi speech sample (22.05kHz mono)
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures & TestClient configuration
│   └── test_api.py          # 11 comprehensive automated tests
├── requirements.txt         # Production dependencies
├── AI_LOG.md                # Comprehensive AI tooling log & decisions
└── README.md                # Project documentation
```

### Key Architectural Highlights
- **Hardware-Optimized CPU Inference**: Uses `faster-whisper` with `int8` quantization and `cpu_threads=4`. Download footprint is only ~145 MB and RAM usage is ~250 MB, perfectly tailored for standard developer laptops.
- **Robust Audio Decoding**: Custom `soundfile` and `librosa` audio loader decodes `.wav` and `.mp3` directly to 16kHz float32 NumPy arrays, bypassing external `ffmpeg` binary dependencies and PyAV version mismatches on Windows.
- **Threadpool Offloading**: CPU-bound ASR execution is dispatched via `asyncio.to_thread` to preserve FastAPI event loop responsiveness.
- **Multilingual Unicode Normalization**: Normalizer strips Western and Indic punctuation (including dandas `।`, `॥`) while strictly preserving Devanagari vowel signs/matras and conjuncts.

---

## 📡 API Reference & Sample Requests/Responses

### 1. `POST /transcribe`
Transcribes audio (`.wav` or `.mp3`) and computes Word Error Rate (WER) if an optional reference transcript is provided.

#### English Clip Sample (`samples/english_sample.wav`)
**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@samples/english_sample.wav" \
  -F "reference=And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
```

**Response (HTTP 201 Created):**
```json
{
  "id": 1,
  "filename": "english_sample.wav",
  "detected_language": "en",
  "language_probability": 0.9449,
  "transcript": "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
  "reference_transcript": "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
  "wer": 0.0,
  "duration_seconds": 11.0,
  "inference_time_seconds": 3.0982,
  "created_at": "2026-11-23 11:21:32"
}
```

#### Hindi Clip Sample (`samples/hindi_sample.wav`)
**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@samples/hindi_sample.wav"
```

**Response (HTTP 201 Created):**
```json
{
  "id": 2,
  "filename": "hindi_sample.wav",
  "detected_language": "hi",
  "language_probability": 0.9274,
  "transcript": "On this video, me, i putting you on youtube and ready for this video, how will we locate Scrap effect ? Despite of that, there is a safety procedures and high post office on screen.",
  "reference_transcript": null,
  "wer": null,
  "duration_seconds": 12.0,
  "inference_time_seconds": 22.747,
  "created_at": "2026-11-23 14:07:02"
}
```

---

### 2. `GET /transcripts`
Lists past transcriptions with pagination (`limit`, `offset`).

**Request:**
```bash
curl "http://127.0.0.1:8000/transcripts?limit=10&offset=0"
```

**Response (HTTP 200 OK):**
```json
[
  {
    "id": 2,
    "filename": "hindi_sample.wav",
    "detected_language": "hi",
    "transcript": "On this video, me, i putting you on youtube...",
    "wer": null,
    "created_at": "2026-11-23 14:07:02"
  },
  {
    "id": 1,
    "filename": "english_sample.wav",
    "detected_language": "en",
    "transcript": "And so my fellow Americans, ask not what your country can do for you...",
    "wer": 0.0,
    "created_at": "2026-11-23 11:21:32"
  }
]
```

---

### 3. `GET /transcripts/{id}`
Returns full record details by ID, or HTTP 404 if not found.

---

### 4. `GET /search?q={query}`
Searches stored transcripts using case-insensitive substring search and returns excerpt snippets.

**Request:**
```bash
curl "http://127.0.0.1:8000/search?q=fellow"
```

**Response (HTTP 200 OK):**
```json
[
  {
    "id": 1,
    "filename": "english_sample.wav",
    "detected_language": "en",
    "transcript": "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
    "match_snippet": "...And so my fellow Americans, ask not ...",
    "created_at": "2026-11-23 11:21:32"
  }
]
```

---

## 🔍 Analytical Note: Why WER Differs Between English and Indian Languages

In speech recognition benchmarks, Indian languages (such as Hindi, Tamil, and Bengali) frequently exhibit significantly higher Word Error Rates (WER) than English when evaluated on multilingual models like Whisper. The root causes and targeted solutions are outlined below:

### 1. Root Causes
1. **Pretraining Data Imbalance**: Whisper’s 680,000-hour pretraining dataset is heavily skewed towards English (>65%), whereas Indian languages constitute less than 1% of the corpus. Consequently, the model's acoustic and language priors are much weaker for Indic phonology.
2. **Devanagari Orthography & Conjunct Consonants (संयुक्ताक्षर)**: Indic scripts are alpha-syllabic (abugidas). A single phonetic error or missing virama (्) merges or splits words into distinct Unicode representations, penalizing word-level Levenshtein distance severely even if the acoustic prediction was almost phonetically accurate.
3. **Schwa Deletion & Dialectal Variations**: In spoken Hindi, the implicit schwa vowel /ə/ is dropped at word ends and between consonants depending on regional dialects (e.g., *karnā* vs *karanā*), leading to frequent spelling discrepancies between hypothesis and ground-truth references.
4. **Subword Tokenizer Fragmentation**: Whisper’s byte-level BPE tokenizer was optimized predominantly on Latin text. In English, common words correspond to 1–2 tokens. In Devanagari, a single character often decomposes into 3–4 UTF-8 byte tokens, increasing the autoregressive sequence length and compounding hallucination/drift risks.
5. **Code-Mixing (Hinglish)**: Colloquial Indian speech frequently alternates between English and Indic vocabulary, causing language classification oscillation and transcription transliteration mismatches.

### 2. Actionable Improvement Strategy
* **Domain-Specific Acoustic Fine-Tuning**: Fine-tune Whisper or employ specialized Indic architectures (such as **AI4Bharat IndicConformer** or **IndicWhisper**) trained on extensive Indian audio corpora (e.g., Kathbath, IndicVoices, FLEURS).
* **Phoneme-Aware & Character Error Rate (CER) Metric Normalization**: Augment evaluation with Character Error Rate (CER) and Grapheme-to-Phoneme (G2P) representations to evaluate phonetic comprehension rather than strict orthographic spelling variations.
* **Indic-Optimized Tokenization**: Train an extended BPE vocabulary natively supporting Indic character clusters and matras to reduce sequence length and decoding latency.

---

## 📋 What Was Not Finished & Future Roadmap

Given the 1-hour build assignment scope, the following improvements are prioritized for production deployment:

1. **Streaming Speech Recognition**: Implement WebSocket endpoint (`/transcribe/stream`) utilizing chunked VAD and audio buffering for live real-time captions.
2. **Asynchronous Background Task Queue**: Integrate Redis and Celery/RQ for processing large audio files (>50 MB) without keeping HTTP client connections open.
3. **Vector Semantic Search**: Upgrade `/search` from substring search to semantic embedding search utilizing `sqlite-vec` or Chroma with an Indic-friendly sentence embedding model (e.g., `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`).
4. **Speaker Diarization**: Integrate PyAnnote.audio to differentiate and tag multiple speakers in conversational audio.
