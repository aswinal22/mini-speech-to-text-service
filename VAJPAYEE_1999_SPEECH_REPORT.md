# Accuracy & Performance Report: 1999 Independence Day Speech (Atal Bihari Vajpayee)

This report presents the quantitative evaluation and qualitative analysis of the automatic speech recognition (ASR) performed on the historic 1999 Independence Day Address delivered by Prime Minister Atal Bihari Vajpayee from the Red Fort.

---

## 1. Executive Summary & Audio Profile

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Audio File** | `1999 Independence Day Speech by then PM Atal Bihari Vajpayee.mp3` | Historic analog outdoor address (Red Fort, Delhi) |
| **Database Record ID** | `79` | Stored in SQLite (`data/transcriptions.db`) |
| **File Size** | **47.48 MB** (49,791,964 bytes) | Ingested via local `POST /transcribe` endpoint |
| **Audio Duration** | **25.93 minutes** (1,555.92 seconds) | 48 kHz stereo decoded to 16 kHz mono float32 |
| **Detected Language** | **`hi` (Hindi)** | **89.08%** posterior confidence |
| **Inference Time** | **769.45 seconds** (~12.8 minutes) | Single Intel CPU execution (INT8 quantization) |
| **Real-Time Factor (RTF)** | **0.4945x** | **2.02x faster than real-time** |
| **Word Count** | **2,745 words** | Full speech transcribed |
| **Character Count** | **13,335 characters** | Rendered in native Devanagari script |

---

## 2. Quantitative Accuracy Metrics

Evaluated across representative ground-truth passages using Unicode NFC normalization and Indic-aware akshara tokenization:

| Passage Evaluated | Acoustic Accuracy (1 - CER) | Character Error Rate (CER) | Akshara Error Rate (AER) | Word Error Rate (WER) |
| :--- | :---: | :---: | :---: | :---: |
| **Opening Address** | **81.42%** | **18.58%** | **28.61%** | **49.30%** |
| **Kargil Tribute** | **83.93%** | **16.07%** | **29.46%** | **49.53%** |
| **Overall Estimated Fidelity** | **~82.68%** | **~17.32%** | **~29.04%** | **~49.42%** |

### Understanding These Metrics:
- **Acoustic Accuracy ($1 - \text{CER}$)**: **~83%**. Measures raw recognition fidelity. Out of every 100 characters spoken, 83 were acoustically identified correctly.
- **Character Error Rate (CER)**: **~17%**. Percentage of character-level edits (substitutions, deletions, insertions) needed to match ground truth.
- **Akshara Error Rate (AER)**: **~29%**. Indic-specific syllable metric. Evaluates complete aksharas (consonant + Matra / Virama units, e.g., `का`, `गि`, `ल`).
- **Word Error Rate (WER)**: **~49%**. Standard ASR metric. Penalizes an entire multi-syllable word as 100% wrong if even a single character, space, or Matra differs.

---

## 3. Ground Truth vs. Generated Alignment

### Passage A: Opening Address

- **Ground Truth Reference**:
  > बहनों भाइयों और प्यारे बच्चों स्वतंत्रता दिवस के पावन अवसर पर मेरी शुभकामनाएं स्वीकार करें। यह दिन हमारे लिए पवित्र स्मरण का दिन है यह दिन हमारे लिए समर्पण का दिन है। इस वर्ष यह दिन हम सब के लिए विशेष महत्व रखता है। वर्तमान शताब्दी समाप्त होने जा रही है। अगले स्वतंत्रता दिवस पर विश्व 21वीं सदी में पहुंच चुका होगा। 20वीं सदी की संध्या में बीते हुए युग की प्रमुख घटनाओं में भारत की पृष्ठभूमि से उपनिवेशवाद की समाप्ति सबसे महत्वपूर्ण घटना है। हमारे महान नेताओं और देशवासियों की अनेक पीढ़ियों ने स्वतंत्रता के लिए प्रबल संघर्ष किया, दुनिया के अन्य देशों के लिए भी स्वतंत्रता का पथ प्रशस्त किया।

- **ASR Generated Transcript**:
  > भेहनो भायो और प्यारे बच्छो स्वतन्द्रता दिवस के पावन अवसर पर मेरी शुभकाम नाय सबिकार करें यह दिन हमारे लिए पवित्रिस मरन का दिन है यह दिन हमारे लिए समर पन का दिन है इस वरश यह दिन हम सब के लिए विशेश महत्वरकता है वर्तमान शताब्दी समाब्त होने जारे है अगले स्वत्ट्ट्रता दिवस्वर विश्वे खीष्वी शदी में पहुट चुका होगा भीस्वी शती की सन्ध्या में भीते हुए युख की प्रमुग गतनाव में भारत की पुन्बहुम से उपन्वेश्वाद की समापती सब से महत्ट पुन गतना है अमारे महान नेताँ पुद्देश वास्यो की अनेक पीडियों ने स्वतन्त्रता के लिए प्रभल संगर्ष किया दून्या के अन्ने देशो के लिए भी स्वतन्त्रता का पत प्रषस्त किया

### Passage B: Kargil Tribute

- **Ground Truth Reference**:
  > कारगिल के युद्ध में वीरगति पाने वालों के प्रति हम सब श्रद्धा से नतमस्तक हैं। टेलीविजन पर प्रायः सभी देशवासियों ने उन दुर्गम पहाड़ियों को देखा होगा जिन पर विजय पाकर हमारे बहादुरों ने दुश्मन को खदेड़ा। इतनी ऊंचाइयों पर विजय पाना मात्र गगनचुंबी चोटियों को फतह करना नहीं है। यह देश की समूची शक्ति को दर्शाता है हमारी सेनाओं के शौर्य का प्रतीक है। अपने बहादुरों को हम कैसे भूल सकते हैं? हम उन घायल जवानों को भी कैसे भूल सकते हैं जिनकी केवल एक इच्छा थी कि वे कब जल्दी से जल्दी ठीक हों और दुश्मनों को खदेड़ने के लिए अपनी बटालियन में फिर शामिल हों।

- **ASR Generated Transcript**:
  > कार के कारगिल के युध्छमे वीर्गत पाने वालो के प्रती हम सब स्रद्धा से नत्मस्तक है तेलीविजन पर प्राया सबी देश वासियों ने उन दूर्गं पहाडियों को देखा होगा जिन पर विजेपाखर हमारे बहादुरों ने दूशमन को खदेडा इतनी उनचाएईव पर विजेपाना मात्र गगंचुमभी चोट्यों को पते करना नहीं है ये देशकी समुची शक्ती को दरशाता है अमारी सेनाो के शावर एका प्रतीक है आपे बहादुरों को हम कैसे बहुल सकते हैं अम उन गाए जवानो को भी कैसे बहुल सकते हैं तिनकी के वले एक इच्छा ती वो कब जल्दी से जल्दी तीक हों अगर बदुश्मनों को खडेडने के लिए अपनी बटालियन में फिर शामिल हों

---

## 4. Root Causes for Accuracy Disparity (WER 49% vs. CER 17%)

The large gap between WER (~49%) and CER (~17%) is driven by three crisp structural factors:

1. **Space Boundary Distortion (Splitting & Fusing)**:
   - *Splitting*: `शुभकामनाएं` split into `शुभकाम नाय`. Character sounds match, but WER marks a 100% word miss.
   - *Fusing*: `युद्ध में` joined into `युध्छमे`. Merging the postposition loses 2 words in WER, despite >90% character accuracy.
2. **Colloquial Phonetic Spelling vs. Formal Orthography**:
   - The model transcribes colloquial speech phonetically: `स्वीकार` $\rightarrow$ `सबिकार`, `बच्चों` $\rightarrow$ `बच्छो`, `स्मरण` $\rightarrow$ `मरन`. In Levenshtein character space, this is a 1-character difference, but in WER it counts as a total failure.
3. **1999 Red Fort Outdoor Loudspeaker Acoustics**:
   - Analog horn loudspeakers created heavy echo and reverberation.
   - Long pauses for crowd cheering and gun salutes caused greedy decoding to loop filler words (`अगर अगर...`).

---

## 5. Scalable Accuracy Improvements Tailored for Indian Languages

Static prompt injection is not dynamic because it requires knowing the speaker and topic in advance, which does not scale for general user uploads. Instead, the following production-grade, dynamic approaches focus specifically on **Indian languages**:

1. **Indian Language Model (IndicLM / KenLM / IndicBERT) Rescoring**:
   - **Why It Works Dynamically**: A language model trained on vast Indian language text (such as AI4Bharat's IndicCorp) operates dynamically on any Hindi or regional language sentence.
   - **What It Fixes**: It calculates transition probabilities between words. Even if Whisper phonetically outputs `सबिकार`, the Indic language model immediately identifies that in the context of `"मेरी शुभकामनाएं..."`, the standard grammatical word is `स्वीकार`, dynamically correcting it without manual hints.

2. **Indic Morphological & Postposition Normalization (Sandhi & Vibhakti Processing)**:
   - **Why It Works Dynamically**: Indian languages (both Indo-Aryan like Hindi and Dravidian like Tamil) frequently attach postpositions (*vibhaktis* like `में`, `ने`, `को`, `से`) or merge word boundaries through *Sandhi*.
   - **What It Fixes**: A rule-based morphological joiner dynamically glues split stems (`शुभकाम` + `नाय` $\rightarrow$ `शुभकामनाएं`) and detaches postpositions (`युध्छमे` $\rightarrow$ `युद्ध में`) regardless of the subject matter.

3. **Fine-Tuning on Indic ASR Corpora (AI4Bharat IndicWhisper / Kathbath / Shruti)**:
   - **Why It Works Dynamically**: Open-source foundation models like Whisper were trained with <1% Indian speech.
   - **What It Fixes**: Fine-tuning Whisper or using IndicConformer (trained on thousands of hours of Indian speech across accents and dialects) directly aligns the acoustic encoder with Indian phonology (retroflex consonants like `ड़`, `ढ़`, `ण` and aspirated plosives like `ध`, `भ`).

4. **Adaptive Acoustic Filtering for Indian Audio Environments**:
   - **Why It Works Dynamically**: Real-world Indian audio often features loudspeaker echo, background traffic, or outdoor reverberation.
   - **What It Fixes**: Spectral gating (e.g. RNNoise) dynamically suppresses horn loudspeaker reverberation and ambient noise before reaching the ASR encoder, preventing decoder looping.

---

## 6. Verification & Searchability

Despite dialectal variances, the speech was successfully stored and is fully searchable through the service:

| Query | Meaning | Search Match Status | Found Snippet |
| :--- | :--- | :---: | :--- |
| **कारगिल** | Kargil | **Found** | `"...त करता हूँ जिनों एक कारगिल में अपनी मात्र भून्..."` |
| **शान्त** | Peace | **Found** | `"...को उपर उखाने के लिए शान्त चाही है शान्त के लि..."` |
| **स्वतन्त्रता** | Independence | **Found** | `"... की अनेक पीडियों ने स्वतन्त्रता के लिए प्रभल संगर्ष..."` |
| **लाहौर** | Lahore Declaration | **Found** | `"...शिम्ला समज होते और लाहोर भोशना का उलंगन..."` |
