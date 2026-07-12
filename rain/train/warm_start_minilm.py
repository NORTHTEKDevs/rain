"""MiniLM warm-start for the codebook.

Phase 1.1 (the strong version). Loads sentence-transformers
all-MiniLM-L6-v2 (~80 MB on disk, 384-dim float embeddings), encodes a
token list, then writes each encoded vector into the codebook via the
existing sign-projection path in warm_start_from_vectors.

Why MiniLM over FastText cc.en.300.bin:
  * 80 MB vs 7 GB
  * Sentence-level semantics work for both chars and BPE subwords
  * No external file download needed beyond the HF cache

For BPE subword codebooks this is the recommended init: each subword is
encoded as a "sentence" by MiniLM, so semantically-related subwords
("inter" vs "intra", "the" vs "a") share hypervector signal after the
sign projection.

NEGATIVE FINDING for char-level codebooks (Tiny Shakespeare A/B,
3K steps, dim=1024):
  warm_start_chars (feature)  -> 2.40 nats/char, within=0.82  cross=0.08
  warm_start_chars_minilm     -> 2.73 nats/char, within=0.48  cross=0.46

MiniLM's single-char embeddings are dominated by its sentence-pooling
artifact, not character semantics, so category separation collapses to
~0.01 and the prior is barely usable. For chars, prefer
warm_start_chars. Keep the MiniLM path for the BPE case via
warm_start_bpe_minilm().

API stays the same as warm_start_from_vectors: pass the resulting dict
into bootstrap_phase1(warm_start_vectors=...) and the existing
sign-tile-store path handles dimension matching.
"""

from __future__ import annotations

import numpy as np

from rain.core.relational import Codebook
from rain.train.warm_start import warm_start_from_vectors

_MODEL_CACHE: dict[str, object] = {}


def _load_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load and cache the sentence-transformers model."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = m
    return m


def encode_tokens(
    tokens: list[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """Encode each token via MiniLM, return {token: float32 vector}."""
    model = _load_model(model_name)
    embs = model.encode(tokens, batch_size=batch_size, show_progress_bar=False)
    return {tok: np.asarray(embs[i], dtype=np.float32) for i, tok in enumerate(tokens)}


def warm_start_chars_minilm(
    codebook: Codebook,
    corpus_text: str,
    *,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Warm-start the codebook for every unique char in the corpus.

    Returns a stats dict mirroring warm_start_chars output so the two
    methods are A/B comparable from the caller.
    """
    chars = sorted(set(corpus_text))
    vectors = encode_tokens(chars, model_name=model_name)
    loaded = warm_start_from_vectors(codebook, vectors)

    # Prior-quality stats: cosine cohesion among letters vs cross-category.
    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float32)
        b = b.astype(np.float32)
        return float(a @ b) / (float(np.linalg.norm(a)) * float(np.linalg.norm(b)) + 1e-9)

    letters = [vectors[c] for c in chars if c.isalpha() and c.islower()]
    digits = [vectors[c] for c in chars if c.isdigit()]
    within_letter = (
        float(np.mean([_cos(a, b) for i, a in enumerate(letters) for b in letters[i + 1 :]]))
        if len(letters) > 1
        else 0.0
    )
    across = (
        float(np.mean([_cos(a, b) for a in letters[:8] for b in digits[:8]]))
        if letters and digits
        else 0.0
    )

    return {
        "chars_seeded": float(loaded),
        "mean_cos_within_letters": within_letter,
        "mean_cos_letters_to_digits": across,
        "embedding_dim": float(next(iter(vectors.values())).shape[0]),
    }


def warm_start_bpe_minilm(
    codebook: Codebook,
    subwords: list[str],
    *,
    model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Warm-start the codebook from a BPE subword vocabulary.

    Useful once the BPE tokenizer is trained and we want subword-aware
    init for HYMN. Returns a stats dict with the same shape as the char
    variant for symmetry.
    """
    vectors = encode_tokens(list(subwords), model_name=model_name)
    loaded = warm_start_from_vectors(codebook, vectors)
    return {
        "tokens_seeded": float(loaded),
        "embedding_dim": float(next(iter(vectors.values())).shape[0]),
    }
