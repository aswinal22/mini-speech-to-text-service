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
curl -X POST "http://127.0.0.1:8000/transcribe" -F "file=@samples/english_1_jfk.wav" -F "reference=And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
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
│       └── evaluator.py     # Multilingual text normalization, WER & CER computation
├── models/small/            # Offline cached faster-whisper small model weights
├── samples/
│   ├── english_1_jfk.wav    # 11.0s JFK speech sample (16kHz mono)
│   ├── english_2_mlk.wav    # 13.0s MLK speech sample (16kHz mono)
│   ├── Audio1.mp3           # 10.7s Tamil speech recording
│   ├── Audio17.mp3          # 10.8s Tamil speech recording
│   └── tts_1795456...mp3    # Hindi speech recordings
├── tests/
│   ├── __init__.py
│   ├── conftest.py          # Pytest fixtures & TestClient configuration
│   └── test_api.py          # 12 comprehensive automated tests
├── BENCHMARK.md             # Ground-truth benchmark report across 10 audio files
├── VAJPAYEE_1999_SPEECH_REPORT.md # Real-world 26-min historical speech benchmark & Indic analysis
├── requirements.txt         # Production dependencies
├── AI_LOG.md                # Comprehensive AI tooling log & decisions
└── README.md                # Project documentation
```

### Key Architectural Highlights
- **Hardware-Optimized CPU Inference**: Uses `faster-whisper` with `int8` quantization and auto-detected CPU threads (`cpu_threads=6`). Download footprint is only ~460 MB and RAM usage is ~450 MB, perfectly tailored for standard developer laptops.
- **Large File & Long-Duration Audio Support**: Expanded upload capacity to **100 MB** (supporting 30–60+ minute audio streams). The decoding pipeline streams normalized 16kHz mono audio directly into RAM, capping peak memory under ~250 MB even on full 26-minute stereo speeches without memory spikes.
- **VAD Chunking for Extended Speeches**: Employs Silero Voice Activity Detection (`vad_filter=True`, `min_silence_duration_ms=500`) to dynamically segment long-duration audio, filter out background applause/silence pauses, and prevent decoder looping during crowd intervals.
- **Robust Audio Decoding**: Custom `soundfile` and `librosa` audio loader decodes `.wav` and `.mp3` directly to 16kHz float32 NumPy arrays, bypassing external `ffmpeg` binary dependencies and PyAV version mismatches on Windows.
- **Threadpool Offloading**: CPU-bound ASR execution is dispatched via `asyncio.to_thread` to preserve FastAPI event loop responsiveness.
- **Multilingual Unicode Normalization**: Normalizer strips Western and Indic punctuation (including dandas `।`, `॥`) while strictly preserving Devanagari/Tamil vowel signs/matras and conjuncts.

---

## 📡 API Reference & Sample Requests/Responses

### 1. `POST /transcribe`
Transcribes audio (`.wav`, `.mp3`, `.m4a`, or `.ogg`) and computes Word Error Rate (WER) and Character Error Rate (CER) if an optional reference transcript is provided.

#### English Clip Sample (`samples/english_1_jfk.wav`)
**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@samples/english_1_jfk.wav" \
  -F "reference=And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."
```

**Response (HTTP 201 Created):**
```json
{
  "id": 1,
  "filename": "english_1_jfk.wav",
  "detected_language": "en",
  "language_probability": 0.9533,
  "transcript": "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
  "reference_transcript": "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
  "wer": 0.0,
  "cer": 0.0,
  "duration_seconds": 11.0,
  "inference_time_seconds": 4.72,
  "created_at": "2026-11-23 23:24:00"
}
```

#### Tamil Clip Sample (`samples/Audio1.mp3`)
**Request:**
```bash
curl -X POST "http://127.0.0.1:8000/transcribe" \
  -F "file=@samples/Audio1.mp3" \
  -F "language=ta"
```

**Response (HTTP 201 Created):**
```json
{
  "id": 2,
  "filename": "Audio1.mp3",
  "detected_language": "ta",
  "language_probability": 0.9287,
  "transcript": "செயற்கை நுண்ணறிவு வேகமாக வளர்ந்து வருகிறது...",
  "reference_transcript": null,
  "wer": null,
  "cer": null,
  "duration_seconds": 10.7,
  "inference_time_seconds": 8.12,
  "created_at": "2026-11-23 23:25:00"
}
```
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
    "filename": "Audio1.mp3",
    "detected_language": "ta",
    "transcript": "செயற்கை நுண்ணறிவு வேகமாக வளர்ந்து வருகிறது...",
    "wer": null,
    "cer": null,
    "created_at": "2026-11-23 23:25:00"
  },
  {
    "id": 1,
    "filename": "english_1_jfk.wav",
    "detected_language": "en",
    "transcript": "And so my fellow Americans, ask not what your country can do for you...",
    "wer": 0.0,
    "cer": 0.0,
    "created_at": "2026-11-23 23:24:00"
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
    "filename": "english_1_jfk.wav",
    "detected_language": "en",
    "transcript": "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country.",
    "match_snippet": "...And so my fellow Americans, ask not ...",
    "created_at": "2026-11-23 11:21:32"
  }
]
```

---

## 📊 Regional Language Accuracy & Benchmarks

Evaluating speech recognition for Indian languages (such as Tamil and Hindi) requires measuring both **Word Error Rate (WER)** and **Character Error Rate (CER)**:
- **WER**: Standard metric for English, but artificially penalizes Indic languages due to agglutination and Sandhi compounding (where grammatical suffixes attach directly to base words).
- **CER**: Evaluates Levenshtein edit distance at the Unicode character level after NFC normalization, reflecting true phonetic accuracy without compounding bias.

Full empirical benchmark results across 10 audio recordings (5 English, 2 Tamil, 3 Hindi) are documented in **[BENCHMARK.md](BENCHMARK.md)**. For real-world long-form evaluation, see the 26-minute historical speech analysis in **[VAJPAYEE_1999_SPEECH_REPORT.md](VAJPAYEE_1999_SPEECH_REPORT.md)**.

---

## 📋 What Was Not Finished & What We Would Do Next

Given the 1-hour build assignment scope, the core working pipeline is fully functional and tested. With more time and infrastructure, the following improvements would be prioritized:

1. **Streaming Speech Recognition**: Implement a WebSocket endpoint (`/transcribe/stream`) with chunked VAD audio buffering for live real-time captioning.
2. **Dedicated Indic ASR Checkpoints**: Drop in regional checkpoints (such as AI4Bharat's IndicWhisper) fine-tuned specifically on Indian dialectal speech.
3. **Vector Semantic Search**: Upgrade `/search` from substring search to semantic embedding search utilizing `sqlite-vec` or Chroma.
4. **Forced Alignment for Subtitle Sync**: Integrate CTC forced alignment to generate word-level timestamped `.srt` files for Same Language Subtitling (SLS).
