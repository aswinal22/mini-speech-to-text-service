import re
import unicodedata
from typing import Dict, List, Optional
import jiwer


def normalize_text(text: str) -> str:
    """
    Multilingual-aware text normalization for fair speech evaluation:
    - Normalizes Unicode to Canonical Composition (NFC).
    - Lowercases Latin characters (English).
    - Fully preserves Indic Unicode letters (category 'L*') and combining
      vowel signs/matras/viramas (category 'M*') across Indian languages.
    - Strips Western and Indic punctuation (category 'P*') and symbols (category 'S*'),
      automatically handling dandas (। ॥), Arabic punctuation (، ؟), etc., without hardcoding.
    - Strips invisible Unicode format control characters (category 'Cf', e.g. ZWNJ U+200C
      and ZWJ U+200D) to avoid spurious mismatches from keyboard ligature formatting.
    - Collapses multiple whitespace into a single space.
    """
    if not text:
        return ""

    # Apply Unicode Canonical Composition (NFC)
    text = unicodedata.normalize("NFC", text)

    chars = []
    for char in text:
        cat = unicodedata.category(char)
        # Drop invisible formatting characters (ZWNJ \u200c, ZWJ \u200d)
        if cat == "Cf":
            continue
        # Replace punctuation and symbols with spaces
        elif cat.startswith(("P", "S")):
            chars.append(" ")
        else:
            chars.append(char)

    normalized = "".join(chars).lower()

    # Collapse repeated whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def tokenize_indic_aksharas(text: str) -> List[str]:
    """
    Tokenizes text into Indic alphasyllabic units (Akshara clusters).
    An Akshara consists of a base consonant/vowel followed by attached vowel matras,
    halants/viramas, or pulli marks (Unicode category 'M*').
    """
    clean_text = normalize_text(text)
    clusters: List[str] = []
    for char in clean_text:
        cat = unicodedata.category(char)
        if cat.startswith("M") and clusters:
            # Combine vowel mark with preceding base consonant
            clusters[-1] += char
        elif char != " ":
            clusters.append(char)
    return clusters


def calculate_wer(hypothesis: str, reference: Optional[str]) -> Optional[float]:
    """
    Compute Word Error Rate (WER) using standardized Levenshtein distance.
    Returns None if reference is None or whitespace only.
    """
    if reference is None:
        return None

    ref_clean = normalize_text(reference)
    hyp_clean = normalize_text(hypothesis)

    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0

    if not hyp_clean:
        return 1.0

    if ref_clean == hyp_clean:
        return 0.0

    wer_value = jiwer.wer(reference=ref_clean, hypothesis=hyp_clean)
    return round(float(wer_value), 4)


def calculate_cer(hypothesis: str, reference: Optional[str]) -> Optional[float]:
    """
    Compute Character Error Rate (CER) at Unicode character level.
    Provides fine-grained acoustic accuracy for Indic languages without
    penalizing Sandhi / word compound splits.
    """
    if reference is None:
        return None

    ref_clean = normalize_text(reference)
    hyp_clean = normalize_text(hypothesis)

    if not ref_clean:
        return 0.0 if not hyp_clean else 1.0

    if not hyp_clean:
        return 1.0

    if ref_clean == hyp_clean:
        return 0.0

    cer_value = jiwer.cer(reference=ref_clean, hypothesis=hyp_clean)
    return round(float(cer_value), 4)


def calculate_aer(hypothesis: str, reference: Optional[str]) -> Optional[float]:
    """
    Compute Akshara Error Rate (AER) across Indic syllabic clusters.
    Evaluates phonetic syllable accuracy (Consonant + Virama + Matra).
    """
    if reference is None:
        return None

    ref_aksharas = " ".join(tokenize_indic_aksharas(reference))
    hyp_aksharas = " ".join(tokenize_indic_aksharas(hypothesis))

    if not ref_aksharas:
        return 0.0 if not hyp_aksharas else 1.0

    if not hyp_aksharas:
        return 1.0

    if ref_aksharas == hyp_aksharas:
        return 0.0

    aer_value = jiwer.wer(reference=ref_aksharas, hypothesis=hyp_aksharas)
    return round(float(aer_value), 4)


def calculate_regional_error_metrics(
    hypothesis: str,
    reference: Optional[str],
    language: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """
    Comprehensive multi-granular error evaluation mechanism for regional languages:
    - wer: Word Error Rate (whitespace delimited)
    - cer: Character Error Rate (Unicode grapheme level)
    - aer: Akshara Error Rate (Syllable cluster level)
    - acoustic_accuracy: 1.0 - min(1.0, CER)
    """
    if reference is None or not reference.strip():
        return {
            "wer": None,
            "cer": None,
            "aer": None,
            "acoustic_accuracy": None,
        }

    wer_val = calculate_wer(hypothesis, reference)
    cer_val = calculate_cer(hypothesis, reference)
    aer_val = calculate_aer(hypothesis, reference)

    acoustic_acc = None
    if cer_val is not None:
        acoustic_acc = round(max(0.0, 1.0 - min(1.0, cer_val)), 4)

    return {
        "wer": wer_val,
        "cer": cer_val,
        "aer": aer_val,
        "acoustic_accuracy": acoustic_acc,
    }
