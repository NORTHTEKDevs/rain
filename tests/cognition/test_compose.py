from rain.cognition.compose import CompositionalReasoner
from rain.core.relational import Codebook


def test_compose_and_extract_roundtrips():
    cb = Codebook(vocab_size=32, dim=2048, seed=0)
    cr = CompositionalReasoner(cb, slot_names=["color", "shape"])
    composite = cr.compose({"color": "red", "shape": "circle"})
    assert composite.shape == (2048,)
    # Extract back
    color = cr.extract(composite, "color", candidates=["red", "blue", "green"])
    shape = cr.extract(composite, "shape", candidates=["circle", "square", "triangle"])
    assert color == "red"
    assert shape == "circle"


def test_compose_unknown_slot_raises():
    cb = Codebook(vocab_size=8, dim=512, seed=0)
    cr = CompositionalReasoner(cb, slot_names=["color"])
    import pytest

    with pytest.raises(ValueError):
        cr.compose({"shape": "x"})
