import re
import unicodedata
from typing import Optional
import jiwer


def normalize_text(text: str) -> str:
    """
    Multilingual-aware text normalization for fair WER calculation:
    - Lowercases Latin characters (English).
    - Fully preserves Indic Unicode letters (category 'L*') and combining
      vowel signs/matras/viramas (category 'M*').
    - Strips Western punctuation (category 'P*'), symbols ('S*'), and Indic dandas (।, ॥).
    - Collapses multiple whitespace into a single space.
    """
    if not text:
        return ""

    # Replace punctuation characters and symbols with spaces, preserving letters and combining marks
    chars = []
    for char in text:
        cat = unicodedata.category(char)
        if cat.startswith(("P", "S")) or char in "।॥":
            chars.append(" ")
        else:
            chars.append(char)

    normalized = "".join(chars).lower()

    # Collapse repeated whitespace
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def calculate_wer(hypothesis: str, reference: Optional[str]) -> Optional[float]:
    """
    Compute Word Error Rate (WER) using standardized Levenshtein distance.
    Returns:
        - None if reference is None or whitespace only.
        - 0.0 if both normalized strings are identical.
        - 1.0 if hypothesis is empty but reference is non-empty.
        - float rounded to 4 decimal places via jiwer.wer.
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
