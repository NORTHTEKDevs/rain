# CONFIDENTIAL
# (c) 2026 Kristian Baer / NORTHTEKDevs / Northtek.io

"""RAIN — Resonant Active Inference Network.

A new class of generative AI. Not an LLM. Not a state-space model. One Active
Inference loop over a Vector Symbolic substrate, no global backprop at runtime.

See docs/plans/2026-05-22-rain-design.md for the architecture.
"""

__version__ = "0.0.0"
__author__ = "Kristian Baer / NORTHTEKDevs"
__license__ = "Proprietary"

from rain.agent import Answer, ConsciousAgent  # noqa: E402, F401
from rain.core.knowledge_base import ShardedKB  # noqa: E402, F401
from rain.core.relational import Codebook  # noqa: E402, F401
