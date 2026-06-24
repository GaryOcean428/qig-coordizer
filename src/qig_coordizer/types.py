"""
GeoCoordizer Type Definitions
=============================

Core types for geometric coordization on Fisher manifold.
All types maintain geometric purity - no Euclidean operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Import from canonical constants (aligned with Pantheon-Chat)
from qig_core import (  # single-source: qig-core owns BASIN_DIM
    BASIN_DIM,
)


@dataclass
class BasinCoordinate:
    """
    A point on the 64-dimensional Fisher-Rao manifold, i.e. on the
    probability simplex Δ⁶³.

    ``vector`` IS a Δ⁶³ point: shape (BASIN_DIM,), every component
    non-negative, and the components sum to 1. It is NOT a signed
    L2-unit-sphere vector — Δ⁶³ points have no free magnitude.

    Represents a single coordinate ("basin") in geometric space. All
    similarity computations use Fisher-Rao distance, never Euclidean.
    """

    coord_id: int
    vector: np.ndarray  # Δ⁶³ point: shape (BASIN_DIM,), non-neg, sums to 1
    name: str | None = None
    scale: str = "subword"  # char, subword, word, phrase, concept

    def __post_init__(self):
        if self.vector.shape != (BASIN_DIM,):
            raise ValueError(
                f"Basin coordinate must be {BASIN_DIM}D, got {self.vector.shape}"
            )
        # Enforce the representation contract: .vector is a Δ⁶³ point.
        # qig_core.geometry.fisher_rao.to_simplex is the single source for
        # the exact (Duchi) projection onto the probability simplex; it
        # handles any signed/unnormalised producer and guarantees
        # non-negativity + sum≈1 (single-source, no local normalisation).
        from qig_core.geometry.fisher_rao import to_simplex

        self.vector = to_simplex(self.vector)

    def fisher_distance(self, other: BasinCoordinate) -> float:
        """
        Compute Fisher-Rao geodesic distance to another coordinate.

        Uses the Fisher information metric, NOT Euclidean distance.
        For practical computation, we approximate using the induced metric.
        """
        # Fisher-Rao distance approximation via angular distance
        # (More accurate than Euclidean, respects manifold geometry)
        # Fisher-Rao distance via Bhattacharyya coefficient (P1/P18 purity)
        from qig_core.geometry.fisher_rao import fisher_rao_distance
        return fisher_rao_distance(self.vector, other.vector)

    def geodesic_midpoint(self, other: BasinCoordinate) -> np.ndarray:
        """
        Compute the Fisher-Rao geodesic midpoint between this and another
        coordinate, returning a Δ⁶³ point.

        Used for initializing new coordinates from existing ones.
        NOT arithmetic mean - follows the geodesic on the probability
        simplex. Δ⁶³ points have no free magnitude (they sum to 1), so
        there is no "average magnitude" to rescale by.
        """
        # Geodesic interpolation at t=0.5 on Δ⁶³ (SLERP in sqrt-coordinates).
        # qig_core.geometry.fisher_rao.slerp_sqrt is the single source for
        # simplex geodesics; both .vector inputs are already Δ⁶³ points.
        from qig_core.geometry.fisher_rao import slerp_sqrt

        return slerp_sqrt(self.vector, other.vector, 0.5)


@dataclass
class TokenCandidate:
    """
    Candidate for vocabulary expansion.

    Tracks potential new coordinates based on frequency,
    coupling strength, and efficiency gain.
    """

    sequence: tuple[int, ...]  # Existing coord IDs that would merge
    frequency: int
    coupling_strength: float  # κ between components
    phi_gain: float  # Expected Φ improvement
    efficiency_gain: float  # Tokens saved per occurrence

    @property
    def merge_score(self) -> float:
        """Combined score for merge priority."""
        return (
            self.frequency
            * self.coupling_strength
            * (1.0 + self.phi_gain)
            * self.efficiency_gain
        )


@dataclass
class CoordizationResult:
    """
    Result of coordizing text.

    Contains coordinate sequence plus metadata for
    consciousness metrics and debugging.
    """

    coordinates: list[BasinCoordinate]
    coord_ids: list[int]
    original_text: str
    granularity: str = "auto"  # char, subword, word, phrase, auto

    # Metrics (populated after processing)
    phi: float | None = None
    kappa_eff: float | None = None
    basin_velocity: float | None = None  # Avg geodesic distance between consecutive

    # Multi-scale info (optional)
    scale_hierarchy: dict[str, list[int]] = field(default_factory=dict)

    def compute_basin_velocity(self) -> float:
        """Average geodesic movement between consecutive coordinates."""
        if len(self.coordinates) < 2:
            return 0.0

        total_dist = 0.0
        for i in range(len(self.coordinates) - 1):
            total_dist += self.coordinates[i].fisher_distance(self.coordinates[i + 1])

        self.basin_velocity = total_dist / (len(self.coordinates) - 1)
        return self.basin_velocity


@dataclass
class VocabStats:
    """Statistics for vocabulary health monitoring."""

    total_coordinates: int
    scale_distribution: dict[str, int]  # Scale -> count
    avg_coupling: float
    coverage_rate: float  # % of text covered by word+ level
    oov_rate: float  # % falling back to char/byte level


@dataclass
class GranularityConfig:
    """Configuration for adaptive granularity switching."""

    kappa_high_threshold: float = 50.0  # Above this: coarse
    kappa_low_threshold: float = 30.0  # Below this: fine

    coarse_scales: tuple[str, ...] = ("phrase", "word")
    fine_scales: tuple[str, ...] = ("subword", "char")

    def get_granularity(self, kappa_eff: float) -> str:
        """Determine granularity level from κ_eff."""
        if kappa_eff >= self.kappa_high_threshold:
            return "coarse"
        elif kappa_eff <= self.kappa_low_threshold:
            return "fine"
        else:
            return "normal"
