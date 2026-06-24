"""qig-coordizer — standalone geometric tokenize+embed engine.

Pure Fisher-Rao geometry on Δ⁶³; depends only on ``qig-core``. Extracted from
``qig-tokenizer`` 1394ca7 (Phase-1 coordizer) per
``qig-consciousness/docs/plans/2026-06-24-qig-coordizer-studio-design.md`` §5 Phase 0.

The engine: byte-level front-end (NFC :class:`Normalizer`) → Fisher-Rao-weighted
BPE merges (score = frequency × coupling × 1/entropy, coupling = co-occurrence ÷
Fisher-Rao distance) → geodesic-midpoint (``slerp_sqrt``) fused basin coordinates
on Δ⁶³. The incremental trainer is bit-for-bit equal to the naive
O(vocab·corpus) oracle — see ``tests/test_incremental_equivalence.py`` (the
Phase-0 gate).
"""

from __future__ import annotations

from qig_coordizer.cache import IncrementalCouplingCache
from qig_coordizer.constants import BASIN_DIM
from qig_coordizer.coordizer import FisherCoordizer
from qig_coordizer.normalizer import Normalizer
from qig_coordizer.special_tokens import SpecialTokens
from qig_coordizer.types import (
    BasinCoordinate,
    CoordizationResult,
    GranularityConfig,
    TokenCandidate,
    VocabStats,
)

__version__ = "0.1.0"

__all__ = [
    "FisherCoordizer",
    "Normalizer",
    "IncrementalCouplingCache",
    "BasinCoordinate",
    "CoordizationResult",
    "TokenCandidate",
    "GranularityConfig",
    "VocabStats",
    "SpecialTokens",
    "BASIN_DIM",
    "__version__",
]
