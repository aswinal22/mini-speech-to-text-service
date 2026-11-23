import requests
import os
import json
from pathlib import Path

files = [
    ("english_1_jfk.wav", "And so, my fellow Americans: ask not what your country can do for you — ask what you can do for your country."),
    ("english_2_mlk.wav", "I have a dream that one day this nation will rise up and live out the true meaning of its creed."),
    ("english_3_librispeech.wav", None),
    ("english_4_librispeech.wav", None),
    ("english_5_ted.wav", None),
    ("tamil_1_talk.wav", None),
    ("tamil_2_talk.wav", None),
    ("hindi_1_speech.wav", None),
    ("hindi_2_lesson.wav", None),
    ("malayalam_1_whisperx.wav", None)
]

results = []
for f, ref in files:
    path = os.path.join("samples", f)
    with open(path, "rb") as fp:
        data = {}
        if ref:
            data["reference_transcript"] = ref
        r = requests.post("http://127.0.0.1:8000/transcribe", files={"file": (f, fp, "audio/wav")}, data=data)
        res = r.json()
        results.append(res)
        lang = res.get("detected_language")
        prob = res.get("language_probability")
        dur = res.get("duration_seconds")
        inf = res.get("inference_time_seconds")
        wer = res.get("wer")
        tx = res.get("transcript")
        print(f"=== {f} ===")
        prob_str = f"{prob:.2%}" if prob is not None else "N/A"
        print(f"  Lang: {lang} ({prob_str}) | Duration: {dur}s | Inference: {inf}s | WER: {wer}")
        print(f"  Transcript: {tx}")

with open("benchmark_results.json", "w", encoding="utf-8") as out:
    json.dump(results, out, ensure_ascii=False, indent=2)

print("\nBenchmark completed and written to benchmark_results.json")
