# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for first- and second-order Theory of Mind."""

from rain.cognition.theory_of_mind import TheoryOfMind


def test_first_order_belief():
    tom = TheoryOfMind(dim=512, num_shards=4, seed=0)
    tom.believe("alice", "marble", "location", "basket")
    assert tom.query_belief("alice", "marble", "location") == "basket"


def test_per_believer_separation():
    """Alice's beliefs don't leak into Bob's KB."""
    tom = TheoryOfMind(dim=512, num_shards=4, seed=0)
    tom.believe("alice", "marble", "location", "basket")
    tom.believe("bob", "marble", "location", "box")
    assert tom.query_belief("alice", "marble", "location") == "basket"
    assert tom.query_belief("bob", "marble", "location") == "box"


def test_unknown_belief_returns_none():
    tom = TheoryOfMind(dim=512, num_shards=4, seed=0)
    assert tom.query_belief("alice", "x", "y") is None


def test_second_order_belief():
    """Alice believes that Bob believes that the marble is in the basket."""
    tom = TheoryOfMind(dim=512, num_shards=4, seed=0)
    tom.believe_second_order("alice", "bob", "marble", "location", "basket")
    assert tom.query_second_order("alice", "bob", "marble", "location") == "basket"


def test_second_order_does_not_leak_to_first_order():
    """A second-order belief about Bob does not change first-order beliefs."""
    tom = TheoryOfMind(dim=512, num_shards=4, seed=0)
    tom.believe("bob", "marble", "location", "box")
    tom.believe_second_order("alice", "bob", "marble", "location", "basket")
    # Bob actually believes "box" — Alice thinks Bob believes "basket"
    assert tom.query_belief("bob", "marble", "location") == "box"
    assert tom.query_second_order("alice", "bob", "marble", "location") == "basket"


def test_believers_listing():
    tom = TheoryOfMind(dim=512, num_shards=4, seed=0)
    tom.believe("alice", "x", "y", "z")
    tom.believe("bob", "x", "y", "z")
    assert sorted(tom.believers()) == ["alice", "bob"]
