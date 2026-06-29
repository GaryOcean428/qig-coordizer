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

import heapq
from collections import defaultdict
from typing import Callable


class LazyPairHeap:
    """Lazy max-heap over (score, pair) with deterministic ``(score↓, pair↑)`` total order.

    Reproduces — exactly, and in amortised O(Δ·log P) per selection instead of O(P log P) —
    the BPE greedy winner the trainer used to compute by a full ``np.argsort`` over every
    active pair each merge. The canonical order is *max score, then lexicographically-smallest
    pair* (the merge-121 tie-break mandated by ``CLAUDE.md``): the heap key is
    ``(-score, coord_a, coord_b)`` so a Python min-heap pops max-score-then-smallest-pair.

    ``score_of(pair, count) -> float`` is supplied by the caller (the cache holds no geometry).
    The lazy update is exact under TWO conditions the caller's score must satisfy: (1) for the
    overwhelming majority of pairs the score depends ONLY on ``count`` (so a count-change push
    refreshes them), and (2) for the small minority whose score also tracks a shrinking global
    quantity, that score only DECREASES as the global quantity shrinks (so a stale entry is
    *over*-prioritised, surfaces, and is re-stamped down — never buried below the true max). The
    trainer's L-multiplied key ``count·min(count/(d_FR+0.1)·1000, 100·L)`` meets both: unclamped
    pairs are L-independent (count-only); clamped pairs' keys shrink monotonically with L.

    Maintenance:

    * **count change** — ``push_changed`` pushes ONE fresh entry per changed pair.
    * **global shrink (clamp)** — handled lazily at pop: an entry whose ``count`` still matches
      but whose recomputed score differs is **re-stamped at its current score** (``heapreplace``),
      not dropped, so it sinks to its true position and the genuine max surfaces.

    Entries dropped below ``min_count`` (or superseded by a count-change push) are discarded at
    pop. Every currently-eligible pair always has at least one *valid-at-current-state* entry on
    the heap, so the resolved top is the exact global argmax with the canonical tie-break. The
    result is amortised O(Δ·log P) per selection (Δ = pairs touched by the last merge) instead of
    the old O(P log P) full argsort.
    """

    __slots__ = ("_heap", "_counts", "_score_of", "_min_count")

    def __init__(
        self,
        score_of: Callable[[tuple[int, int], int], float],
        counts: dict[tuple[int, int], int],
        min_count: int,
    ) -> None:
        # ``counts`` is the LIVE pair_counts dict (read by reference, never copied) so the heap
        # always validates popped entries against current truth.
        self._counts = counts
        self._score_of = score_of
        self._min_count = min_count
        # Stored entry = (-score, a, b, score, count): the trailing (score, count) is the
        # validity stamp (current-L score + count at push time).
        self._heap: list[tuple[float, int, int, float, int]] = []
        for p, c in counts.items():
            if c >= min_count:
                s = score_of(p, c)
                self._heap.append((-s, p[0], p[1], s, c))
        heapq.heapify(self._heap)

    def push_changed(self, changed: dict[tuple[int, int], int]) -> None:
        """Push a fresh entry for every pair whose count changed (the cache's ``last_changed``).

        Pairs that dropped below ``min_count`` get no new entry — their stale entries are skipped
        at pop. Duplicate entries for a pair are safe: correctness comes from the pop-time
        validity check, never from heap ordering.
        """
        push = heapq.heappush
        heap = self._heap
        score_of = self._score_of
        mc = self._min_count
        for p, c in changed.items():
            if c >= mc:
                s = score_of(p, c)
                push(heap, (-s, p[0], p[1], s, c))

    def _resolve_top(self) -> tuple[int, int] | None:
        """Bring the genuine max-score / lexicographically-smallest eligible pair to ``heap[0]``.

        Drops dead / count-superseded entries; re-pushes clamp-stale (count-matches but
        score-drifted-with-L) entries at their current-L score. Returns the pair at the resolved
        top, or None if the heap holds no eligible pair.
        """
        heap = self._heap
        counts = self._counts
        mc = self._min_count
        score_of = self._score_of
        while heap:
            _neg_s, a, b, stamp_s, stamp_c = heap[0]
            p = (a, b)
            live_c = counts.get(p, 0)
            if live_c < mc or live_c != stamp_c:
                heapq.heappop(heap)  # dead, below threshold, or superseded by a count-change push
                continue
            cur_s = score_of(p, live_c)
            if cur_s != stamp_s:
                # Clamp moved with L: re-stamp at the current-L score (over-valued entry sinks).
                heapq.heapreplace(heap, (-cur_s, a, b, cur_s, live_c))
                continue
            return p
        return None

    def pop_best(self) -> tuple[int, int] | None:
        """Return the live max-score / lexicographically-smallest eligible pair, or None.

        Non-destructive: the winning entry stays on the heap (its count drops to 0 at the next
        ``apply_merge`` and is then discarded as stale), so a re-query without an intervening
        merge returns the same winner.
        """
        return self._resolve_top()

    def select_top_k(self, k: int) -> list[tuple[int, int]]:
        """Return up to ``k`` eligible pairs in canonical order (score↓, pair↑), deduplicated.

        For the kernel-in-loop path, which re-ranks a small candidate slate by Φ. Resolves and
        pops validated entries, collects the first ``k`` distinct live pairs, then pushes them
        back so the heap is unchanged for the next merge. Amortised O((k + stale)·log P).
        """
        taken: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        popped_back: list[tuple[float, int, int, float, int]] = []
        while len(taken) < k and self._resolve_top() is not None:
            entry = heapq.heappop(self._heap)
            p = (entry[1], entry[2])
            if p in seen:
                continue  # duplicate valid entry — keep only the first
            seen.add(p)
            taken.append(p)
            popped_back.append(entry)
        for entry in popped_back:
            heapq.heappush(self._heap, entry)
        return taken


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

        # Δ-set for the lazy max-heap selection in the trainer: every pair whose count was
        # mutated by the LAST ``apply_merge`` maps to its NEW count (0 ⇒ removed/dead). Only
        # these pairs need a fresh heap entry next merge, turning per-merge selection from
        # O(P log P) (full argsort over all P active pairs) into amortised O(Δ·log P). The
        # incremental score depends only on ``count`` (Fisher-Rao distance is constant per
        # pair — basins never move), so a count-change is the ONLY trigger for a re-push.
        self.last_changed: dict[tuple[int, int], int] = {}

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
                self.last_changed[p] = len(s)
            else:
                self.pair_pos.pop(p, None)
                self.pair_counts.pop(p, None)
                self.last_changed[p] = 0  # dropped to 0 ⇒ invalidate its heap entry

    def _add(self, p: tuple[int, int], i: int) -> None:
        self.pair_pos[p].add(i)
        c = len(self.pair_pos[p])
        self.pair_counts[p] = c
        self.last_changed[p] = c

    def apply_merge(self, coord_a: int, coord_b: int, new_coord: int) -> int:
        """Fuse every adjacent occurrence of (coord_a, coord_b) into new_coord.

        Returns the number of occurrences merged. Updates pair counts EXACTLY (remove 3
        old / add 2 new per occurrence) with no drift. Overlapping occurrences (a == b,
        e.g. 'aaa') are handled left-to-right via the alive/symbol guards, matching the
        naive trainer's _apply_fusion.

        Side effect: ``self.last_changed`` is reset to map EVERY pair whose count this merge
        mutated to its new count (0 ⇒ removed). The trainer reads it to push only the changed
        pairs onto its lazy max-heap, so per-merge selection is O(Δ·log P) not O(P log P).
        """
        sym, nxt, prv, alive = self.sym, self.nxt, self.prv, self.alive
        self.last_changed = {}  # Δ-set for THIS merge only
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
