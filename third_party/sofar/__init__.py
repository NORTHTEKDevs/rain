# Copyright 2026 Kristian Baer
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SOFAR (Signal-Optimized Frequency-Aligned Routing)
"""SOFAR: bio-inspired acoustic geometry routing for transformers."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from sofar.mapper import (
    ChannelMap,
    LayerChannels,
    available_scoring_formulas,
    map_channels,
    score_channel,
)
from sofar.encoder import (
    BandGate,
    EntropyAdaptiveGate,
    FrequencyBands,
    FrequencyLayeredEncoder,
    encode_bands,
    fuse_bands,
)
from sofar.attention import (
    BeamMode,
    BeamSteeringAdapter,
    patch,
    select_beam_mode,
    unpatch,
)
from sofar.training import (
    CalibrationExample,
    TrainingConfig,
    TrainingResult,
    load_adapter_checkpoint,
    make_synthetic_corpus,
    save_adapter_checkpoint,
    train_adapter,
)
from sofar.corpora import (
    make_code_corpus,
    make_legal_corpus,
    make_medical_corpus,
    make_mixed_corpus,
)
from sofar.export import export_bundle, from_pretrained

# Single source of truth: pyproject.toml [project].version. Falls back
# to a sentinel when running from an uninstalled checkout (no metadata
# resolvable).
try:
    __version__ = _pkg_version("sofar")
except PackageNotFoundError:
    __version__ = "0.0.0+source"
__all__ = [
    # mapper
    "ChannelMap",
    "LayerChannels",
    "available_scoring_formulas",
    "map_channels",
    "score_channel",
    # encoder
    "FrequencyLayeredEncoder",
    "FrequencyBands",
    "BandGate",
    "EntropyAdaptiveGate",
    "encode_bands",
    "fuse_bands",
    # attention
    "BeamSteeringAdapter",
    "BeamMode",
    "select_beam_mode",
    "patch",
    "unpatch",
    # training
    "CalibrationExample",
    "TrainingConfig",
    "TrainingResult",
    "make_synthetic_corpus",
    "train_adapter",
    "save_adapter_checkpoint",
    "load_adapter_checkpoint",
    # corpora
    "make_legal_corpus",
    "make_medical_corpus",
    "make_code_corpus",
    "make_mixed_corpus",
    # bundle export / import
    "export_bundle",
    "from_pretrained",
]
