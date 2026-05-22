# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""BPE tokenizer via SentencePiece. Default vocab 32K, configurable."""

from __future__ import annotations
import io
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

    def encode(self, text: str) -> list[int]:
        assert self._sp is not None, "tokenizer not trained"
        return self._sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        assert self._sp is not None, "tokenizer not trained"
        return self._sp.decode(ids)

    def save(self, path: str | Path) -> None:
        assert self._sp is not None
        Path(path).write_bytes(self._sp.serialized_model_proto())

    def load(self, path: str | Path) -> None:
        self._sp = spm.SentencePieceProcessor(model_file=str(path))
