# AI Usage & Engineering Log (`AI_LOG.md`)

This document records the AI tooling, prompts, errors encountered, corrections made, and independent human decisions during the implementation of the PlanetRead Mini Speech-to-Text Service.

---

## 1. Which AI Tools Were Used, and For What?

* **Google Antigravity & Gemini 3.8**:
  * **System Architecture & Planning**: Auditing hardware resources (RAM, CPU threads, absence of ffmpeg on Windows host) and formulating the 7-stage subtask execution plan.
  * **Code Generation**: Scaffolding FastAPI routes, SQLite database layer, Pydantic schemas, and pytest integration test cases.
  * **Debugging & Root-Cause Analysis**: Diagnosing C-extension DLL paths, regex Unicode script edge cases in Devanagari, and PyAV version incompatibilities.

---

## 2. The 3–5 Most Important Prompts (Copied as Written)

### Prompt 1: Initial Task Ingestion & Planning
> `/plan This is the task which we are gonna build`  
> `I need a complete comprehesive plan to do that.`  
> `This is the repo link https://github.com/aswinal22/mini-speech-to-text-service.git we can skip the docker part and we need to breakdown the complex task into step by step working subtasks`

### Prompt 2: Local Hardware Constraint Enforcement
> `Hey Make sure that these requirements meets the hardware which i have cause I gonna run that locally`

### Prompt 3: Architectural Rigor (Full Class & Function Signatures)
> `make sure to add the functions/classes each of the files that gonna consists of with its signature so that it will be easy while generating the code`

### Prompt 4: Step-by-Step Execution Approval
> `Approved. Make sure to process step by step and without completing the each of the steps dont jump`

### Prompt 5: Real-World Local Verification
> `Hey we can handle the bechmarking step later now run that locally and I need to test whether all the components are working?`

---

## 3. Places Where the AI Was Wrong or Unhelpful, How We Noticed, and How We Fixed It

### Incident A: Unicode Regex Stripping Devanagari Combining Marks (Matras)
* **What the AI Suggested**: The AI initially used a standard text normalization regex:
  ```python
  normalized = re.sub(r"[^\w\s]", " ", text)
  ```
* **How We Noticed**: During unit testing with Indian language text (`नमस्ते, दुनिया।`), the normalized output was unexpectedly mangled into:
  ```
  नमस त द न य
  ```
  Upon investigating character categories via `unicodedata.category()`, we discovered that in Python's standard `re` module, `\w` matches category `L` (Letters) and `N` (Numbers), but rejects `Mn` (Nonspacing Marks). In Devanagari, vowel signs (*matras*) and viramas (्, े, ु, ी, ा) belong to `Mn`. The AI's regex stripped every matra from the text, invalidating any Hindi Word Error Rate calculation.
* **How We Fixed It**: Overrode the naive regex with an explicit Unicode category filter:
  ```python
  chars = [
      " " if unicodedata.category(c).startswith(("P", "S")) or c in "।॥" else c
      for c in text
  ]
  normalized = re.sub(r"\s+", " ", "".join(chars)).strip().lower()
  ```
  This preserves both letters and combining vowel signs while stripping Western and Indic punctuation. The output cleanly restored to `नमस्ते दुनिया`.

---

### Incident B: PyAV (`av.open`) `metadata_errors` Breaking Audio Decoding
* **What the AI Suggested**: The AI assumed that `faster-whisper.transcribe("file.wav")` would decode audio without issues using the installed `av` (PyAV) wheel.
* **How We Noticed**: Running transcription on the downloaded JFK `.wav` sample triggered an immediate crash:
  ```
  TypeError: open() got an unexpected keyword argument 'metadata_errors'
  ```
  The pip dependency resolver had installed `av==19.0.2` (the latest major release), which deprecated and removed the `metadata_errors` parameter that `faster_whisper/audio.py` passes to `av.open`.
* **How We Fixed It**: Rather than relying on `faster-whisper`'s internal file decoding path, we implemented an audio waveform preprocessor using `soundfile` and `librosa` (`load_audio_waveform` in `app/services/asr.py`). It reads the audio file and resamples it directly into a mono 16kHz `float32` NumPy array. Passing NumPy arrays directly to `model.transcribe(waveform)` completely bypassed the buggy `av.open` call and eliminated any dependency on system `ffmpeg`.

---

### Incident C: Windows C-Extension DLL Path Resolution
* **What the AI Suggested**: The AI initially assumed standard `import faster_whisper` would work out of the box after running `pip install`.
* **How We Noticed**: On Python 3.8+ on Windows, importing wheels with bundled DLLs (such as `av.libs` and `ctranslate2.libs`) raised `ModuleNotFoundError: No module named 'av'` because Windows security policies restrict implicit DLL search paths.
* **How We Fixed It**: Injected a cross-platform DLL loader into `app/__init__.py`:
  ```python
  if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
      for sp in [p for p in sys.path if "site-packages" in p]:
          for lib_dir in Path(sp).glob("*.libs"):
              try:
                  os.add_dll_directory(str(lib_dir))
              except Exception:
                  pass
  ```
  This guaranteed all native C++ dependencies load cleanly across both test harnesses and running servers.

---

## 4. Engineering Decision Made Independently Rather Than Taking the AI's Default Suggestion

### Decision: Constraining Execution to INT8 Quantization on CPU (`compute_type="int8"`) with Explicit Core Allocation (`cpu_threads=4`)

* **What Default AI Would Do**: The default AI template recommended using standard `openai-whisper` with `torch` in `float32` precision or multi-worker Uvicorn setups.
* **Why We Overrode It**:
  1. **Memory Budget**: Live host telemetry revealed only **~1.1 GB of available physical RAM** out of 7.65 GB total on the host machine. A standard PyTorch `float32` Whisper model consumes 1.5–3.0 GB of memory, which would trigger severe Windows virtual memory swapping or an immediate Out-Of-Memory (OOM) crash.
  2. **Quantization & Efficiency**: By enforcing `faster-whisper` with `compute_type="int8"`, memory consumption dropped to $\approx 250\text{ MB}$, fitting comfortably within the 1.1 GB headroom.
  3. **Concurrency Control**: Setting `cpu_threads=4` on the 8-thread Intel CPU ensured that background speech recognition leaves 4 logical cores uninhibited for the operating system and FastAPI's HTTP server event loop.
