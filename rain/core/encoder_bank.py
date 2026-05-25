# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Encoder bank: every modality -> the same hypervector space.

The architectural commitment of RAIN-Net is that text, images, audio,
code, and numeric data all live in the same D=10K bipolar HV space.
This module is the single entry point that enforces that contract.

For any modality m and any payload p, encoder_bank.encode(m, p) returns
a (D,) float32 array that is comparable via cosine similarity with HVs
produced for any other modality. Multi-modal queries reduce to ordinary
HV ops: bind(text_query_hv, image_input_hv) gives a multi-modal query
HV that you can match against a multi-modal KB.

LLMs handle multi-modality by training a separate encoder per modality
(CLIP-style) and a fusion layer in the transformer. RAIN-Net handles
it via the substrate. Drop a new modality by writing ONE encoder; the
rest of the system stays unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rain.core.hv_substrate import (
    DEFAULT_DIM,
    Codebook,
    bind,
    bundle,
    hash_to_hv,
    permute,
)
from rain.core.multimodal_kb import (
    encode_audio_spectrogram,
    encode_image_patches,
    encode_numeric_timeseries,
)


def _word_to_hv_ngram(word: str, dim: int) -> np.ndarray:
    """Encode a single word via character bigram bundling.

    Words sharing substrings produce similar HVs (eg. 'apple' and 'apples'
    share 4 of 5 bigrams), which gives the text encoder semantic
    robustness without training. Pure whitespace+hash encoding produces
    totally distinct HVs for these morphological variants, which makes
    KB retrieval fragile.

    Encoding:
        bigrams = ['#a', 'ap', 'pp', 'pl', 'le', 'e#']    # word-bounded
        word_hv = bundle(hash_to_hv(b1), permute(hash_to_hv(b2), 1), ...)

    Position-bound so 'ab' != 'ba' but still allows some commutativity
    through the bundle.
    """
    if not word:
        return hash_to_hv("word::<empty>", dim=dim)
    padded = f"#{word.lower()}#"
    bigrams = [padded[i : i + 2] for i in range(len(padded) - 1)]
    if not bigrams:
        return hash_to_hv(f"word::{word}", dim=dim)
    parts: list[np.ndarray] = []
    for i, bg in enumerate(bigrams[:32]):
        parts.append(permute(hash_to_hv(f"bg::{bg}", dim=dim), i))
    return bundle(*parts)


_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "in", "on", "at", "to", "for", "by", "with", "from", "into",
        "and", "or", "but", "not", "no", "yes", "as", "if", "then", "than",
        "this", "that", "these", "those", "it", "its", "he", "she", "they",
        "we", "you", "i", "his", "her", "their", "our", "your", "my",
        "what", "who", "when", "where", "why", "how", "which", "whom",
        "does", "do", "did", "done", "have", "has", "had", "will", "would",
        "can", "could", "should", "may", "might", "must", "shall",
        "about", "after", "again", "against", "all", "am", "any",
        "because", "before", "below", "between", "both", "during",
        "each", "few", "more", "most", "other", "some", "such", "only",
        "own", "same", "so", "very", "just", "also", "very",
    }
)


def _text_to_hv(text: str, dim: int, codebook: Codebook) -> np.ndarray:
    """Encode a text string into a (dim,) HV that respects content words.

    Pipeline:
        1. Lowercase + strip
        2. Tokenize on word boundaries
        3. Strip stopwords (improves signal-to-noise; common words add
           shared bigrams that pollute orthogonality)
        4. For each content word, build an n-gram HV (morphological/typo
           robustness)
        5. Bundle them WITHOUT position binding (bag of content words);
           position binding hurt synthetic-benchmark accuracy by ~30%
           because paraphrased queries reorder content words

    Measured on a 200-fact synthetic benchmark: this encoder beats
    Jaccard lexical baseline at top-3 and top-5.
    """
    import re

    text_clean = text.strip().lower()
    if not text_clean:
        return hash_to_hv("text::<empty>", dim=dim)
    tokens = [t for t in re.findall(r"[a-z0-9]+", text_clean) if t]
    # Strip stopwords. Keep numbers (e.g. "1969") as they are content.
    content = [t for t in tokens if t not in _STOPWORDS and len(t) >= 2]
    # If everything was a stopword (unusual), fall back to all tokens.
    if not content:
        content = tokens[:32] if tokens else ["empty"]
    word_hvs = [_word_to_hv_ngram(t, dim) for t in content[:64]]
    for t, h in zip(content[:64], word_hvs, strict=True):
        if codebook._items is not None and t not in codebook._items:
            codebook._items[t] = len(codebook._items)
            if codebook._vectors is None or codebook._vectors.size == 0:
                codebook._vectors = h[np.newaxis, :].astype(np.float32)
            else:
                codebook._vectors = np.vstack([codebook._vectors, h[np.newaxis, :]])
    # Bag-of-content-words: order-invariant. Paraphrases share content words.
    return bundle(*word_hvs)


def _code_to_hv(source: str, dim: int, codebook: Codebook) -> np.ndarray:
    """Encode source code. Lightweight: tokenize on punctuation + whitespace,
    add an extra 'code' role binding so code HVs cluster apart from prose.
    """
    import re

    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[(){}\[\];=+\-*/<>.,:]|[0-9]+", source)
    if not tokens:
        return hash_to_hv("code::<empty>", dim=dim)
    code_role = hash_to_hv("modality::code", dim=dim)
    token_hvs = [codebook.add(f"code::{t}") for t in tokens[:256]]
    seq = bundle(*[bind(h, permute(code_role, k)) for k, h in enumerate(token_hvs)])
    return bind(seq, code_role)


def _ts_to_hv(values: np.ndarray, dim: int, max_steps: int) -> np.ndarray:
    """Wrap encode_numeric_timeseries with modality role binding."""
    base = encode_numeric_timeseries(np.asarray(values), dim, max_steps=max_steps)
    role = hash_to_hv("modality::ts", dim=dim)
    return bind(base, role)


def _image_to_hv(image: np.ndarray, dim: int, n_patches: int) -> np.ndarray:
    base = encode_image_patches(image, dim, n_patches=n_patches)
    role = hash_to_hv("modality::image", dim=dim)
    return bind(base, role)


def _audio_to_hv(spectrogram: np.ndarray, dim: int) -> np.ndarray:
    base = encode_audio_spectrogram(spectrogram, dim)
    role = hash_to_hv("modality::audio", dim=dim)
    return bind(base, role)


@dataclass
class EncoderBank:
    """Single entry point for encoding any modality into the shared HV space.

    Modality registry is open: register('lidar', fn) lets you plug in a
    new encoder without touching the rest of the system. The contract is
    that fn(payload, dim) returns a (dim,) float32 array.
    """

    dim: int = DEFAULT_DIM
    codebook: Codebook = field(init=False)
    _custom: dict[str, Callable[[Any, int], np.ndarray]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.codebook = Codebook(dim=self.dim, seed=0)

    def encode(self, modality: str, payload: Any, **kwargs: Any) -> np.ndarray:
        """Encode payload of given modality into a (dim,) HV.

        Built-in modalities: text, code, image, audio, ts.
        Custom modalities registered via register().
        """
        m = modality.lower()
        if m == "text":
            return _text_to_hv(str(payload), self.dim, self.codebook)
        if m == "code":
            return _code_to_hv(str(payload), self.dim, self.codebook)
        if m == "image":
            return _image_to_hv(np.asarray(payload), self.dim, kwargs.get("n_patches", 16))
        if m == "audio":
            return _audio_to_hv(np.asarray(payload), self.dim)
        if m == "ts":
            return _ts_to_hv(np.asarray(payload), self.dim, kwargs.get("max_steps", 64))
        if m in self._custom:
            return self._custom[m](payload, self.dim)
        raise ValueError(f"unknown modality: {modality!r}; register() it or use built-in")

    def encode_multi(self, parts: list[tuple[str, Any]]) -> np.ndarray:
        """Encode a multi-modal compound query.

        Example:
            bank.encode_multi([
                ('text', 'what is in this picture'),
                ('image', img_array),
            ])

        Returns the bind of the parts (multi-modal binding -- the
        result is a single HV that is similar to KB entries matching
        BOTH constraints).
        """
        if not parts:
            raise ValueError("empty parts list")
        hvs = [self.encode(m, p) for m, p in parts]
        if len(hvs) == 1:
            return hvs[0]
        out = hvs[0]
        for h in hvs[1:]:
            out = bind(out, h)
        return out

    def register(self, modality: str, fn: Callable[[Any, int], np.ndarray]) -> None:
        """Register a new modality encoder.

        fn signature: fn(payload, dim) -> (dim,) float32 array.
        """
        self._custom[modality.lower()] = fn

    def list_modalities(self) -> list[str]:
        """All currently registered modality names."""
        return ["text", "code", "image", "audio", "ts"] + list(self._custom.keys())
