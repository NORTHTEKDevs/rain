from raincg.bench.transformer_baseline import (
    build_vocab, Seq2SeqTransformer, greedy_decode,
)


def test_build_vocab_has_specials():
    v = build_vocab([("copy A B", ["A", "B"])])
    for s in ("<pad>", "<bos>", "<eos>", "<unk>"):
        assert s in v.stoi
    assert v.stoi["A"] != v.stoi["B"]


def test_model_forward_and_decode_shapes():
    v = build_vocab([("copy A B", ["A", "B"])])
    m = Seq2SeqTransformer(len(v.stoi), d_model=32, nhead=2, num_layers=1, ff=64)
    out = greedy_decode(m, v, "copy A B", max_len=8)
    assert isinstance(out, list)
    assert all(isinstance(t, str) for t in out)
