"""Reserved special-token scheme for qig-coordizer.

The byte-level coordizer assigns coord ids ``0..255`` to raw bytes and ``256+``
to learned geodesic-fusion merges (see :class:`coordizer.FisherCoordizer`).
Special/control tokens therefore need ids that a byte can never produce.

HONEST SCOPE — NOT YET WIRED. These ids are *defined* here but the live
``FisherCoordizer`` is still pure byte+merge and starts learned merges at id
256. This module is the designated home for the PAD/BOS/EOS the Qwen language
boundary will need (design Phase 3: output-distribution → Δ⁶³ → QIGRAM).
Wiring is deferred to that phase precisely so Phase 0 keeps the
``incremental==naive`` equivalence gate byte-for-byte intact. When wired, the
first learned-merge id must shift past the reserved block (see
:attr:`SpecialTokens.first_merge_id`).
"""

from __future__ import annotations

from dataclasses import dataclass

BYTE_VOCAB_SIZE = 256


@dataclass(frozen=True)
class SpecialTokens:
    """Proposed reserved coordizer ids for boundary/control tokens.

    Provisional layout — not consumed by ``FisherCoordizer`` yet.
    """

    pad: int = 256
    bos: int = 257
    eos: int = 258
    unk: int = 259

    @property
    def count(self) -> int:
        return 4

    @property
    def first_merge_id(self) -> int:
        """First id available for learned geodesic-fusion merges once special
        tokens are wired (currently the live coordizer ignores this and starts
        merges at ``BYTE_VOCAB_SIZE``)."""
        return BYTE_VOCAB_SIZE + self.count


DEFAULT_SPECIAL_TOKENS = SpecialTokens()

__all__ = ["SpecialTokens", "DEFAULT_SPECIAL_TOKENS", "BYTE_VOCAB_SIZE"]
