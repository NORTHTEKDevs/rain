# CONFIDENTIAL - PATENT PENDING
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
from rain.cognition.dialogue import DialogueContext


def test_remember_and_resolve():
    d = DialogueContext()
    d.remember(entity="rome", relation="capital_of")
    e, r = d.resolve_references(None, None)
    assert e == "rome" and r == "capital_of"


def test_partial_resolve_keeps_explicit():
    d = DialogueContext()
    d.remember(entity="rome", relation="capital_of")
    e, r = d.resolve_references("paris", None)
    assert e == "paris" and r == "capital_of"


def test_history_records_turns():
    d = DialogueContext()
    d.record_turn("user", "where is rome?")
    d.record_turn("rain", "italy")
    assert len(d.history) == 2
    assert d.history[0] == ("user", "where is rome?")


def test_reset_clears_state():
    d = DialogueContext()
    d.remember(entity="x", relation="y")
    d.record_turn("user", "hi")
    d.reset()
    assert d.last_entity is None
    assert d.last_relation is None
    assert d.history == []
