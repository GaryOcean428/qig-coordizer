"""Reserved special-token scheme for qig-coordizer.

The byte-level coordizer assigns coord ids 0..255 to raw bytes and 256+ to learned
geodesic-fusion merges (``FisherCoordizer`` starts merges at id 256). Special/control
tokens therefore must NOT occupy the byte range OR the merge range — they are
**appended ABOVE the trained vocabulary**, so they never collide and merges still start
at 256 unchanged.

HONEST SCOPE — NOT YET WIRED. ``FisherCoordizer`` is still pure byte+merge; these ids
are the designated home for the PAD/BOS/EOS the Qwen language boundary will need
(design Phase 3). Wiring is deferred. This scheme is collision-free BY CONSTRUCTION —
the prior draft placed ``pad`` at id 256, which COLLIDED with merge-id 256 (caught in
the council red-team, fixed here).
"""

from __future__ import annotations

from dataclasses import dataclass

BYTE_VOCAB_SIZE = 256


@dataclass(frozen=True)
class SpecialTokens:
    """Reserved control-token ids, appended ABOVE the trained vocab (collision-free).

    Construct with the trained ``vocab_size`` as ``base``; ids become ``base, base+1, …``
    — above every byte (0..255) and every learned merge (256..vocab_size-1). There is no
    safe hardcoded base (it depends on how many merges were learned), so ``base`` is
    required and validated.
    """

    base: int
    names: tuple[str, ...] = ("<pad>", "<bos>", "<eos>", "<unk>")

    def __post_init__(self) -> None:
        if self.base < BYTE_VOCAB_SIZE:
            raise ValueError(
                f"special-token base ({self.base}) must be ≥ {BYTE_VOCAB_SIZE} and at or "
                "above the trained vocab size — special tokens append ABOVE the vocab, "
                "never inside the byte/merge range"
            )

    @property
    def ids(self) -> dict[str, int]:
        return {name: self.base + i for i, name in enumerate(self.names)}

    @property
    def count(self) -> int:
        return len(self.names)


__all__ = ["SpecialTokens", "BYTE_VOCAB_SIZE"]
