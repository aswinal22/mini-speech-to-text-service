# PlanetRead ASR Benchmark & Accuracy Evaluation Report

## 1. Executive Summary & Objective

This benchmark evaluates the **Mini Speech-to-Text Service** across a test suite of **10 audio recordings** in Indian and global languages:
- **5 English** audio samples (`english_1` through `english_5`): Historic speeches, LibriSpeech audiobooks, and conversational TED talks.
- **2 Tamil** audio samples (`Audio1.mp3`, `Audio17.mp3`): Synthetic and natural Dravidian speech describing technology and daily life.
- **3 Hindi** audio samples (`tts_1795456041632`, `tts_1795456143603`, `tts_1795456188137`): Text-to-speech recordings covering customer greeting, philosophy, and motivational quotes.

All benchmarks were evaluated directly against **ground-truth reference transcripts** using local **CPU execution with INT8 quantization** (`faster-whisper`).

---

## 2. Evaluation Methodology: WER & CER Against Ground Truth

In speech recognition research and production evaluations, accuracy is measured using two complementary, standardized metrics:

### 2.1 Word Error Rate (WER)
$$\text{WER} = \frac{S + D + I}{N}$$
Where $S$ is substitutions, $D$ is deletions, $I$ is insertions, and $N$ is the total number of words in the reference transcript.
- **Application**: The standard benchmark for analytical languages with whitespace-delimited words (such as English).
- **Limitation for Indic**: In agglutinative languages (e.g. Tamil `மேம்படுத்துகின்றன`), a single compound split or vowel sign causes an entire multi-syllable word to count as a 100% error, artificially inflating the error rate.

### 2.2 Character Error Rate (CER)
$$\text{CER} = \frac{S_c + D_c + I_c}{N_c}$$
Measures Levenshtein edit distance at the Unicode character level after NFC normalization (preserving Indic Combining Vowel Marks / Matras `Mn` and Viramas).
- **Application**: The primary metric for Indian and morphologically rich languages. Accurately reflects true acoustic and phonetic fidelity without being penalized by compound word boundaries.

---

## 3. Comprehensive Ground-Truth Benchmark Results

| # | Audio Sample | Language | Word Error Rate (WER) | Character Error Rate (CER) | Notes / Result |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | `english_1_jfk.wav` | `en` | **0.00%** | **0.00%** | Exact match |
| 2 | `english_2_mlk.wav` | `en` | **0.00%** | **0.00%** | Exact match |
| 3 | `english_3_librispeech.wav` | `en` | **0.00%** | **0.00%** | Exact match |
| 4 | `english_4_librispeech.wav` | `en` | **0.00%** | **0.00%** | Exact match |
| 5 | `english_5_ted.wav` | `en` | **0.00%** | **0.00%** | Exact match |
| 6 | `Audio1.mp3` | `ta` | **5.26%** | **1.72%** | High precision match (<2% CER) |
| 7 | `Audio17.mp3` | `ta` | 94.44% | 34.73% | Dialectal variation |
| 8 | `tts_1795456041632.mp3` | `hi` | **20.00%** | **3.90%** | High precision match (<4% CER) |
| 9 | `tts_1795456143603.mp3` | `hi` | 25.00% | **5.88%** | Low character error (<6% CER) |
| 10 | `tts_1795456188137.mp3` | `hi` | 33.33% | **9.09%** | Low character error (<10% CER) |

---

## 4. Key Factors Affecting Regional Language Accuracy

English achieved 0.00% error rates, while Tamil and Hindi exhibited higher error rates due to three core factors:

- **Low Training Data Volume**: Foundation models like Whisper are pre-trained on over 65% English audio, while Indian languages account for less than 1% of the training dataset.
- **Word Compounding (Sandhi & Agglutination)**: In Tamil and Hindi, suffixes attach directly to base words. Standard WER penalizes compound boundary variations as full word errors, whereas Character Error Rate (CER) accurately captures the true phonetic fidelity.
- **Code-Switching**: Spoken Indian speech frequently weaves English loanwords (e.g., "AI Voice Assistant", "Text-to-Speech") into native grammar, causing occasional script transitions.

---

## 5. Conclusion & Future Improvements

### Key Takeaways
- **WER vs. CER in Indian Languages**: Standard WER works cleanly for English (0.00%), but penalizes Indic languages due to word compounding (Sandhi). Character Error Rate (CER) gives the truest representation of phonetic accuracy (<10% on most Indic audio, reaching 1.72% on Tamil and 3.90% on Hindi).
- **CPU Efficiency**: Faster-whisper INT8 runs reliably on local consumer hardware without GPU requirements.

### Next Steps with More Time & Resources
- **Dedicated Indic Models & GPU**: Leverage GPUs to run larger regional models (e.g., AI4Bharat IndicWhisper or Whisper-large-v3) with full beam search at sub-second latencies.
- **Phonetic Post-Correction**: Add a lightweight Indic spell-corrector to cleanly resolve retroflex consonants and code-switched loanwords.
- **Live Streaming ASR**: Implement WebSocket streaming with Voice Activity Detection (VAD) for live subtitling.
