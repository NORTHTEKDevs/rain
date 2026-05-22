# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

import pytest
from rain.tokenize.bpe import BPETokenizer

def test_train_and_encode():
    tok = BPETokenizer(vocab_size=512)
    tok.train(["the quick brown fox", "the lazy dog"])
    ids = tok.encode("the fox")
    assert len(ids) > 0
    assert all(isinstance(i, int) for i in ids)

def test_round_trip():
    tok = BPETokenizer(vocab_size=512)
    tok.train(["hello world", "goodbye world"])
    text = "hello world"
    ids = tok.encode(text)
    assert tok.decode(ids).strip() == text
