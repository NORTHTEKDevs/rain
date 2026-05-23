# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Feature-based char-level codebook warm-start.

The design plan calls for warm-starting the HYMN codebook from
FastText/sentence-transformers projections. Those are ~80 MB - 7 GB
external deps. This module ships a smaller-scope alternative that
needs zero downloads: hand-engineered char features projected to
bipolar dim-D vectors.

Features per ASCII character:
  * unicode category (letter / digit / punctuation / whitespace / other)
  * case (lower / upper / non-letter)
  * vowel vs consonant
  * ASCII numeric value (normalized)
  * frequency in the corpus (normalized)

The full feature vector is projected to (D,) via a deterministic random
projection, then sign()ed to a bipolar codebook entry. Chars with similar
features (e.g., all uppercase letters) end up with hypervectors that have
non-trivial cosine similarity, giving HYMN a meaningful prior to refine
instead of pure noise.

Empirical claim: warm-started training reaches the same NLL plateau in
~30-50% fewer steps. To be measured.
"""

from __future__ import annotations

import unicodedata
from collections import Counter

import numpy as np

from rain.core.relational import Codebook

_VOWELS = set("aeiouAEIOU")


def _char_features(char: str, corpus_freq: float) -> np.ndarray:
    """Compute a fixed-length feature vector for one character.

    Order matters: changing it invalidates the projection. New features
    should be appended.
    """
    cat = unicodedata.category(char) if char else "Zs"  # first char only
    code = ord(char) if char else 0
    is_letter = float(cat.startswith("L"))
    is_lower = float(cat == "Ll")
    is_upper = float(cat == "Lu")
    is_digit = float(cat.startswith("N"))
    is_punct = float(cat.startswith("P"))
    is_space = float(cat in ("Zs", "Cc"))
    is_vowel = float(char in _VOWELS)
    is_consonant = float(is_letter and not is_vowel and char.isalpha())
    ascii_norm = (code & 0x7F) / 128.0
    high_byte = float(code > 127)
    return np.array(
        [is_letter, is_lower, is_upper, is_digit, is_punct, is_space,
         is_vowel, is_consonant, ascii_norm, high_byte, corpus_freq],
        dtype=np.float32,
    )


_FEATURE_DIM = 11


def _projection_matrix(seed: int, target_dim: int) -> np.ndarray:
    """Deterministic (FEATURE_DIM, D) projection. Same seed -> same matrix.

    Use a (sparse-ish) Gaussian random projection. Bipolarization happens
    downstream via sign().
    """
    rng = np.random.default_rng(seed)
    return rng.standard_normal((_FEATURE_DIM, target_dim)).astype(np.float32)


def warm_start_chars(
    codebook: Codebook,
    corpus_text: str,
    *,
    seed: int = 7777,
) -> dict[str, int]:
    """Overwrite each codebook entry for each char in `corpus_text` with a
    feature-based vector.

    Returns a stats dict (chars seen, mean cosine within letters, mean cosine
    across categories).
    """
    chars = list(corpus_text)
    freq = Counter(chars)
    total = max(1, len(chars))
    proj = _projection_matrix(seed=seed, target_dim=codebook.dim)

    feature_vecs: dict[str, np.ndarray] = {}
    for c in sorted(freq):
        f = _char_features(c, freq[c] / total)
        projected = (f @ proj).astype(np.float32)
        # Bipolar via sign; +1 for ties so the vector is well-defined.
        bipolar = np.where(projected >= 0, 1, -1).astype(np.int16)
        feature_vecs[c] = bipolar

    # Inject into the codebook cache directly (matches Codebook's internal API).
    for c, v in feature_vecs.items():
        codebook._cache[c] = v

    # Compute a quick prior-quality stat: mean cosine within letter category
    # vs across categories. Higher within / lower across = better prior.
    def _cos(a, b):
        a = a.astype(np.float32); b = b.astype(np.float32)
        return float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

    letters = [v for c, v in feature_vecs.items() if c.isalpha() and c.islower()]
    digits = [v for c, v in feature_vecs.items() if c.isdigit()]
    within_letter = (
        np.mean([_cos(a, b) for i, a in enumerate(letters)
                 for b in letters[i + 1:]]) if len(letters) > 1 else 0.0
    )
    if letters and digits:
        across = float(np.mean([_cos(a, b) for a in letters[:8] for b in digits[:8]]))
    else:
        across = 0.0

    return {
        "chars_seeded": len(feature_vecs),
        "mean_cos_within_letters": float(within_letter),
        "mean_cos_letters_to_digits": across,
    }
