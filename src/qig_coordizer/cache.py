"""IncrementalCouplingCache — the ONE correct incremental adjacent-pair tracker.

Design: docs/20260624-qig-coordizer-stack-design-1.00W.md §3.3 + Appendix C Phase 1.

This REPLACES the deleted, broken ``IncrementalPairStats`` (trainer.py:70-166), whose
``apply_merge`` left ``broken_left``/``broken_right`` as dead variables — the counts of
pairs *broken* by a merge were never decremented, so pair counts drifted upward after
every merge (the live ``CoordinzerTrainer`` was selecting merges off wrong counts), and
whose ``score()`` referenced attributes that did not exist on the class.

The correct mechanism is the doubly-linked-list splice proven bit-for-bit equal to the
naive O(vocab·corpus) oracle in ``FisherCoordizer._train_incremental``
(tests/test_incremental_equivalence.py): per merged occurrence, remove the THREE old pairs
that touched it ``(a,b)``, ``(prev,a)``, ``(b,next)`` and add the TWO new pairs
``(prev,new)``, ``(new,next)``. Per-merge cost is O(occurrences), not O(corpus).

Scope: this tracks adjacency COUNTS only. It deliberately does NOT hold basin vectors,
Fisher-Rao distances, or the merge score — the trainer owns scoring (kernel Φ/κ, or the
frequency×coupling regime the design calls "coupling"). A pair's Fisher-Rao distance is
immutable once both tokens exist (basins never move), so that ``(a,b)→d_FR`` cache lives
with the scorer; here we only maintain the mutable adjacency *counts*.
"""

from __future__ import annotations

from collections import defaultdict


class IncrementalCouplingCache:
    """Exact, drift-free incremental adjacent-pair tracker over a doubly-linked list.

    Drop-in for the old ``IncrementalPairStats`` interface used by ``CoordinzerTrainer``:
    ``__init__(corpus_coords, context_window)``, ``.pair_counts``, ``.get_pairs(min_count)``,
    ``.corpus_len``, ``.corpus_coords``, ``.apply_merge(a, b, new)``. The trainer decides
    WHICH pair to merge; this cache keeps the counts exact in O(occurrences) per merge.
    """

    def __init__(self, corpus_coords: list[int], context_window: int = 3) -> None:
        n = len(corpus_coords)
        self.context_window = context_window
        # Doubly-linked list over the corpus slots (same structure as the proven trainer).
        self.sym: list[int] = list(corpus_coords)           # symbol at each slot
        self.nxt: list[int] = list(range(1, n)) + [-1]      # successor (-1 = end)
        self.prv: list[int] = [-1] + list(range(0, n - 1))  # predecessor
        self.alive = bytearray([1]) * n                     # slot still present?

        # pair -> set of left-slot positions; counts derive from the set sizes.
        self.pair_pos: dict[tuple[int, int], set[int]] = defaultdict(set)
        for i in range(n - 1):
            self.pair_pos[(self.sym[i], self.sym[i + 1])].add(i)
        self.pair_counts: dict[tuple[int, int], int] = {
            p: len(s) for p, s in self.pair_pos.items()
        }

        self._n_alive = n
        # Slot 0 can never be the right half of a merge (j = nxt[i] ≥ 1), so it never dies;
        # walking ``nxt`` from slot 0 visits exactly the alive slots in order.
        self._head = 0 if n else -1

    # -- read interface --------------------------------------------------------------
    @property
    def corpus_len(self) -> int:
        return self._n_alive

    @property
    def corpus_coords(self) -> list[int]:
        """Reconstruct the full live coordinate sequence (O(corpus); needed once at save)."""
        return self.sample(self._n_alive)

    def sample(self, n: int) -> list[int]:
        """First ``n`` live coordinates (O(n)); cheap path for per-merge kernel sampling."""
        out: list[int] = []
        i = self._head
        # Defensive: slot 0 never dies by construction (j = nxt[i] ≥ 1, so slot 0 is never the
        # right half of a merge), but advance past any dead leading slot so reconstruction stays
        # correct even if that invariant is ever changed by future edits.
        while i != -1 and not self.alive[i]:
            i = self.nxt[i]
        while i != -1 and len(out) < n:
            if self.alive[i]:
                out.append(self.sym[i])
            i = self.nxt[i]
        return out

    def get_pairs(self, min_count: int = 5) -> dict[tuple[int, int], int]:
        return {p: c for p, c in self.pair_counts.items() if c >= min_count}

    # -- mutation --------------------------------------------------------------------
    def _remove(self, p: tuple[int, int], i: int) -> None:
        s = self.pair_pos.get(p)
        if s and i in s:
            s.discard(i)
            if s:
                self.pair_counts[p] = len(s)
            else:
                self.pair_pos.pop(p, None)
                self.pair_counts.pop(p, None)

    def _add(self, p: tuple[int, int], i: int) -> None:
        self.pair_pos[p].add(i)
        self.pair_counts[p] = len(self.pair_pos[p])

    def apply_merge(self, coord_a: int, coord_b: int, new_coord: int) -> int:
        """Fuse every adjacent occurrence of (coord_a, coord_b) into new_coord.

        Returns the number of occurrences merged. Updates pair counts EXACTLY (remove 3
        old / add 2 new per occurrence) with no drift. Overlapping occurrences (a == b,
        e.g. 'aaa') are handled left-to-right via the alive/symbol guards, matching the
        naive trainer's _apply_fusion.
        """
        sym, nxt, prv, alive = self.sym, self.nxt, self.prv, self.alive
        merged = 0
        # snapshot positions left-to-right so overlapping occurrences merge like the naive path
        for i in sorted(self.pair_pos.get((coord_a, coord_b), ())):
            if not alive[i] or sym[i] != coord_a:
                continue
            j = nxt[i]
            if j == -1 or not alive[j] or sym[j] != coord_b:
                continue
            h, k = prv[i], nxt[j]
            # remove the three old pairs touching this occurrence
            self._remove((coord_a, coord_b), i)
            if h != -1 and alive[h]:
                self._remove((sym[h], coord_a), h)
            if k != -1 and alive[k]:
                self._remove((coord_b, sym[k]), j)
            # splice: slot i becomes new_coord, slot j is dropped
            sym[i] = new_coord
            nxt[i] = k
            if k != -1:
                prv[k] = i
            alive[j] = 0
            self._n_alive -= 1
            merged += 1
            # add the two new pairs
            if h != -1 and alive[h]:
                self._add((sym[h], new_coord), h)
            if k != -1 and alive[k]:
                self._add((new_coord, sym[k]), i)
        return merged
