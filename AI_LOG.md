# AI Usage & Engineering Log (`AI_LOG.md`)

This document records the AI tooling, prompts, errors encountered, corrections made, and independent human decisions during the implementation of the PlanetRead Mini Speech-to-Text Service.

---

## 1. Which AI Tools Were Used, and For What?

* **Google Antigravity & Gemini 3.8**:
  * **System Architecture & Planning**: Auditing hardware resources (RAM, CPU cores, absence of external ffmpeg binary) and formulating the modular subtask execution plan.
  * **Code Generation**: Scaffolding FastAPI routes, SQLite database layer, Pydantic schemas, and pytest integration test cases.
  * **Debugging & Root-Cause Analysis**: Diagnosing PyAV version incompatibilities, Unicode script normalization edge cases in Indic languages, and SQLite schema migrations.

---

## 2. The 3 Most Important Prompts (Copied as Written)

### Prompt 1: Initial Task Ingestion & Planning
> `/plan This is the task which we are gonna build`  
> `I need a complete comprehesive plan to do that.`  
> `This is the repo link https://github.com/aswinal22/mini-speech-to-text-service.git we can skip the docker part and we need to breakdown the complex task into step by step working subtasks`

### Prompt 2: Architectural Rigor (Full Class & Function Signatures)
> `make sure to add the functions/classes each of the files that gonna consists of with its signature so that it will be easy while generating the code`

### Prompt 3: Regional Language Error Metric Formulation
> `How can we design an accurate error evaluation mechanism tailored for Indian regional languages transcription (like Tamil and Hindi) where standard Word Error Rate fails due to compound words?`

---

## 3. Places Where the AI Was Wrong or Unhelpful, How We Noticed, and How We Fixed It

### Incident A: Unicode Regex Stripping Devanagari & Indic Combining Marks (Matras)
* **What the AI Suggested**: The AI initially proposed a standard text normalization regex:
  ```python
  normalized = re.sub(r"[^\w\s]", " ", text)
  ```
* **How We Noticed**: Testing on Indian language text (`नमस्ते, दुनिया।`) mangled the string into `नमस त द न य`. In Python's standard `re` module, `\w` matches category `L` (Letters) and `N` (Numbers), but rejects `Mn` (Nonspacing Marks). In Indic abugida scripts (Devanagari, Tamil), vowel signs (*matras*) and viramas (्, े, ு, ்) belong to `Mn`. The AI's regex stripped every matra from the text, invalidating Word Error Rate calculations.
* **How We Fixed It**: Implemented an explicit Unicode category filter with NFC canonical composition:
  ```python
  text = unicodedata.normalize("NFC", text)
  chars = []
  for char in text:
      cat = unicodedata.category(char)
      if cat == "Cf":  # Strip invisible ZWNJ / ZWJ formatting codes
          continue
      elif cat.startswith(("P", "S")):  # Replace punctuation & symbols with space
          chars.append(" ")
      else:
          chars.append(char)  # Retain all Letters (L*) and Combining Marks (M*)

  normalized = re.sub(r"\s+", " ", "".join(chars).lower()).strip()
  ```
  This preserves both letters and combining vowel signs while stripping Western and Indic punctuation (including dandas `।`, `॥`), cleanly restoring `नमस्ते दुनिया`.

---

### Incident B: PyAV (`av.open`) `metadata_errors` Breaking `faster-whisper` Decoding
* **What the AI Suggested**: The AI assumed that calling `model.transcribe("file.wav")` in `faster-whisper` would decode audio without issues using the installed `av` (PyAV) wheel.
* **How We Noticed**: Passing audio file paths to `faster-whisper` crashed immediately on execution:
  ```
  TypeError: open() got an unexpected keyword argument 'metadata_errors'
  ```
  The pip dependency resolver had installed `av==19.0.2` (the latest major release), which deprecated and removed the `metadata_errors` keyword argument that `faster-whisper`'s internal `decode_audio` passed to `av.open`.
* **How We Fixed It**: Rather than relying on `faster-whisper`'s internal file decoding, we built a dedicated audio waveform preprocessor using `soundfile` and `librosa` (`load_audio_waveform` in `app/services/asr.py`). It reads the audio file and resamples it directly into a normalized mono 16kHz `float32` NumPy array. Passing the NumPy array directly to `model.transcribe(waveform)` completely bypassed the broken `av.open` call and eliminated external `ffmpeg` binary dependencies.

---

### Incident C: Naive Word Error Rate (WER) Masking Regional Language Acoustic Accuracy
* **What the AI Suggested**: The AI initially treated standard whitespace-delimited Word Error Rate (WER) as the sole accuracy evaluation metric for all languages, adhering strictly to naive ASR benchmarking.
* **How We Noticed**: When evaluating our Tamil speech sample (`Audio1.mp3`), the transcript was phonetically sound (capturing the underlying sentence with near-total intelligibility), yet the AI reported a severely inflated **57.89% Word Error Rate (WER)**. 
  Upon linguistic analysis, we caught that in Dravidian languages (like Tamil), postpositions and verbal suffixes attach directly to root words through **agglutination and Sandhi compounding** (e.g. `மேம்படுத்துகின்றன` vs hypothesis `மேம் பட்டுக்கின்றன`). Standard whitespace WER penalizes any compound split as a 100% word error (1 Substitution + 1 Insertion on a single word), creating a false impression that the model had completely failed.
* **How We Fixed It**: Overrode the AI's naive reliance on WER by engineering a multi-granular evaluation suite integrating **Character Error Rate (CER)**. By evaluating edit distance across normalized Unicode graphemes, the measured error dropped to **21.26% (and 1.72% with domain prompt)**, providing a faithful, linguistically grounded reflection of actual regional speech recognition accuracy.

---

## 4. Engineering Decision Made Independently Rather Than Taking the AI's Default Suggestion

### Decision: Building a Pure-Python In-Memory Audio Pipeline to Eliminate Docker and External `ffmpeg` Binaries

* **What Default AI Would Do**: The AI repeatedly suggested either requiring Docker or instructing the user to download external `ffmpeg.exe` binaries and manually configure Windows system `PATH` variables.
* **Why We Overrode It**:
  1. **Zero-Setup Portability**: The 1-hour build assignment scope prioritized clean, fast evaluation by reviewers. Requiring external system binaries or Docker on Windows introduces permission issues, path conflicts, and unnecessary setup friction.
  2. **In-Memory Waveform Processing**: We chose to implement an in-memory audio decoding pipeline using `soundfile` and `librosa` with peak loudness normalization (0.95). It loads `.wav` and `.mp3` files directly into 16kHz float32 NumPy arrays, scaling low-volume speech and passing raw arrays directly to Whisper.
  3. **Self-Contained Deployment**: This architectural decision made the entire service 100% self-contained, portable, and runnable with a single `pip install -r requirements.txt` without needing administrative privileges or external system dependencies.
