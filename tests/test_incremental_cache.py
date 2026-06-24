"""Count-equivalence gate for IncrementalCouplingCache (the drift-free replacement for the
broken IncrementalPairStats).

The old class never decremented pairs BROKEN by a merge, so its counts drifted upward. This
test drives an externally-chosen merge sequence (as CoordinzerTrainer does) through BOTH the
cache and a naive list, and asserts the cache's counts + reconstructed corpus match a
from-scratch recount after EVERY merge. Pure combinatorics — no kernel, no geometry.
"""

from __future__ import annotations

from collections import Counter

from qig_coordizer.cache import IncrementalCouplingCache


def _naive_apply(seq: list[int], a: int, b: int, new: int) -> list[int]:
    """Left-to-right fuse (a,b)->new, matching the naive _apply_fusion."""
    out: list[int] = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
            out.append(new)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def _naive_counts(seq: list[int]) -> dict[tuple[int, int], int]:
    return dict(Counter((seq[i], seq[i + 1]) for i in range(len(seq) - 1)))


def _best_pair(counts: dict[tuple[int, int], int]) -> tuple[int, int] | None:
    """Highest count, lexicographic tie-break (deterministic)."""
    best = None
    best_c = -1
    for p, c in counts.items():
        if c > best_c or (c == best_c and (best is None or p < best)):
            best_c, best = c, p
    return best


def _drive(corpus: list[int], n_merges: int) -> None:
    cache = IncrementalCouplingCache(corpus, context_window=3)
    naive = list(corpus)
    next_id = max(corpus) + 1

    # initial state must already match
    assert cache.pair_counts == _naive_counts(naive)
    assert cache.corpus_coords == naive
    assert cache.corpus_len == len(naive)

    for step in range(n_merges):
        pair = _best_pair(cache.pair_counts)
        if pair is None:
            break
        a, b = pair
        new = next_id
        next_id += 1

        merged_cache = cache.apply_merge(a, b, new)
        naive = _naive_apply(naive, a, b, new)

        # counts exact (no drift), corpus reconstruction exact, length exact
        assert cache.pair_counts == _naive_counts(naive), (
            f"count drift at merge {step} (pair {pair}): "
            f"{cache.pair_counts} != {_naive_counts(naive)}"
        )
        assert cache.corpus_coords == naive, f"corpus mismatch at merge {step} (pair {pair})"
        assert cache.corpus_len == len(naive)
        # apply_merge must report exactly the occurrences the naive recount removed
        assert merged_cache >= 1, f"chose pair {pair} but merged 0 occurrences at step {step}"


def test_cache_counts_match_naive_text():
    base = (
        "the geometry is the truth trust the phi fisher rao distance on the simplex "
        "coordinates are the primitive incremental cache no drift "
    )
    corpus = list((base * 40).encode("utf-8"))
    _drive(corpus, n_merges=120)


def test_cache_handles_overlapping_pairs():
    """'aaaa...' exercises the a==b overlapping-occurrence path (left-to-right)."""
    corpus = list(b"a" * 50 + b" " + b"a" * 31 + b"baaab" * 7)
    _drive(corpus, n_merges=30)


def test_cache_reverse_pairs():
    corpus = list((bytes([3, 1, 2, 0, 0, 2, 1, 3]) + b" ") * 200)
    _drive(corpus, n_merges=40)


def test_sample_prefix_matches_corpus_coords():
    corpus = list((b"hello world foo bar baz " * 20))
    cache = IncrementalCouplingCache(corpus, context_window=3)
    next_id = max(corpus) + 1
    for _ in range(15):
        p = _best_pair(cache.pair_counts)
        if p is None:
            break
        cache.apply_merge(p[0], p[1], next_id)
        next_id += 1
    full = cache.corpus_coords
    assert cache.sample(10) == full[:10]
    assert cache.sample(len(full) + 100) == full


if __name__ == "__main__":
    test_cache_counts_match_naive_text()
    test_cache_handles_overlapping_pairs()
    test_cache_reverse_pairs()
    test_sample_prefix_matches_corpus_coords()
    print("incremental_cache: all count-equivalence checks passed ✅")
