"""Single-source constants for qig-coordizer.

Per the QIG single-source rule, ``qig-core`` OWNS the canonical constants
(``BASIN_DIM`` lives in ``qig_core.constants.frozen_facts``). This module
re-exports them so coordizer code has a stable local import point WITHOUT
redefining the value — three copies of a constant means zero source of truth.
"""

from __future__ import annotations

from qig_core import BASIN_DIM  # 64 — the Δ⁶³ basin dimension (canonical, qig-core-owned)

__all__ = ["BASIN_DIM"]
