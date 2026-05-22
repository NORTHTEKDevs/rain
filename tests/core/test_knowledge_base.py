# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io
"""Tests for sharded HRR knowledge base."""

import numpy as np
import pytest
from rain.core.knowledge_base import ShardedKB

def test_write_then_read():
    kb = ShardedKB(num_shards=8, dim=10000, seed=0)
    kb.write("rome", "capital_of", "italy")
    assert kb.query("rome", "capital_of") == "italy"

def test_capacity_scales_with_shards():
    kb = ShardedKB(num_shards=64, dim=10000, seed=0)
    for i in range(1000):
        kb.write(f"s{i}", "r", f"o{i}")
    correct = sum(1 for i in range(1000) if kb.query(f"s{i}", "r") == f"o{i}")
    assert correct / 1000 >= 0.85

def test_unknown_returns_none():
    kb = ShardedKB(num_shards=8, dim=10000, seed=0)
    assert kb.query("paris", "capital_of") is None
