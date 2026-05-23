# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""BPE tokenizer via SentencePiece. Default vocab 32K, configurable."""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import sentencepiece as spm


class BPETokenizer:
    def __init__(self, vocab_size: int = 32000) -> None:
        self.vocab_size = vocab_size
        self._sp: spm.SentencePieceProcessor | None = None

    def train(self, texts: list[str]) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for t in texts:
                f.write(t + "\n")
            path = f.name
        try:
            model_buf = io.BytesIO()
            spm.SentencePieceTrainer.train(
                input=path,
                model_writer=model_buf,
                vocab_size=self.vocab_size,
                model_type="bpe",
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                hard_vocab_limit=False,
            )
            self._sp = spm.SentencePieceProcessor(model_proto=model_buf.getvalue())
        finally:
            os.unlink(path)

    def encode(self, text: str) -> list[int]:
        if self._sp is None:
            raise RuntimeError("tokenizer not trained -- call .train() or .load() first")
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        if self._sp is None:
            raise RuntimeError("tokenizer not trained -- call .train() or .load() first")
        return self._sp.decode(ids)

    def save(self, path: str | Path) -> None:
        if self._sp is None:
            raise RuntimeError("tokenizer not trained -- call .train() or .load() first")
        Path(path).write_bytes(self._sp.serialized_model_proto())

    def load(self, path: str | Path) -> None:
        self._sp = spm.SentencePieceProcessor(model_file=str(path))
