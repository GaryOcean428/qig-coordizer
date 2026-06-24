"""Equivalence gate: FisherCoordizer._train_incremental MUST equal _train_naive.

This is the Phase-1 binary verifier for the qig-coordizer stack
(docs/20260624-qig-coordizer-stack-design-1.00W.md §3.3, Appendix C Phase 1).

The incremental doubly-linked-list trainer with the write-once (a,b)->d_FR
`fisher_cache` is the optimised production path; `_train_naive` is the O(vocab*corpus)
correctness oracle. They MUST produce a bit-for-bit identical merge sequence and
identical fused basin vectors on a small corpus (byte -> up to 4k vocab), otherwise
the optimisation has silently changed the vocabulary.

Run directly for a timed report:
    .venv/bin/python tests/test_incremental_equivalence.py
or under pytest:
    .venv/bin/python -m pytest tests/test_incremental_equivalence.py -q
"""

from __future__ import annotations

import time

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # allow standalone `python tests/test_incremental_equivalence.py`
    pytest = None

from qig_coordizer import FisherCoordizer


def _corpus(target_bytes: int = 8000) -> bytes:
    """Deterministic ~target_bytes corpus with rich, repeated structure so that
    many adjacent pairs clear min_pair_count (gives the trainers real merges to make)."""
    base = (
        "the geometry is the truth trust the phi. fisher rao distance on the simplex "
        "is the unique markov invariant metric. coordinates are the primitive not an "
        "afterthought. text goes in basin coordinates come out in one pass. the "
        "incremental cache recomputes only what changed the naive oracle rescans the "
        "whole corpus. consciousness equals information geometry. the manifold needs "
        "depth which equals abstraction token phrase sentence document. "
    )
    reps = (target_bytes // len(base)) + 1
    return (base * reps)[:target_bytes].encode("utf-8")


def _unicode_corpus(target_bytes: int = 8000) -> bytes:
    """Multibyte-heavy corpus: every char emits 1-4 UTF-8 bytes, exercising the
    full 0-255 byte range (curly quotes, accents, em-dash, Greek, CJK). The
    *trainer* must be byte-for-byte equivalent on these too — this is the byte
    soup the Normalizer front-end will later normalise, but equivalence of the
    incremental cache vs naive oracle must already hold on raw bytes."""
    base = (
        "café — naïve coördination résumé. “smart quotes” and ‘apostrophes’. "
        "Δ⁶³ basins on the Fisher–Rao manifold; range π/2. 量子情報幾何 と 意識。 "
        "geometry is the truth — trust the φ. αβγ coupling, κ attractor. "
    )
    reps = (target_bytes // len(base.encode("utf-8"))) + 1
    return (base * reps).encode("utf-8")[:target_bytes]


def _reverse_pair_corpus(target_bytes: int = 6000) -> bytes:
    """Adversarial corpus engineered to provoke reverse-pair ties where the
    lexicographically-LARGER pair (b,a) tends to occur BEFORE the smaller (a,b).

    Pre-fix, the naive oracle broke ties by first-occurrence (dict order) while
    the incremental path broke ties lexicographically — so a corpus where (b,a)
    appears first would diverge. With the unified lexicographic tie-break both
    paths must now agree. Built from a palindromic block so reverse pairs occur
    with near-equal counts and symmetric contexts (the entropy/coupling tie)."""
    # bytes chosen so the reverse pair is lexicographically inverted vs occurrence
    block = bytes([3, 1, 2, 0, 0, 2, 1, 3]) + b" "  # 3..1..2..0 then mirror
    reps = (target_bytes // len(block)) + 1
    return (block * reps)[:target_bytes]


def _train(naive: bool, target_vocab: int, corpus: bytes, min_pair_count: int = 2) -> FisherCoordizer:
    c = FisherCoordizer(target_vocab_size=target_vocab)
    c.train(
        corpus,
        context_window=5,
        min_pair_count=min_pair_count,
        verbose=False,
        naive=naive,
    )
    return c


def _assert_equivalent(target_vocab: int, corpus: bytes) -> tuple[int, float, float]:
    """Train both paths, assert identical merges + vectors. Returns (n_merges, t_inc, t_naive)."""
    t0 = time.perf_counter()
    inc = _train(False, target_vocab, corpus)
    t_inc = time.perf_counter() - t0

    t0 = time.perf_counter()
    nai = _train(True, target_vocab, corpus)
    t_naive = time.perf_counter() - t0

    # 1. Identical merge sequence (the core claim).
    assert inc.merge_rules == nai.merge_rules, (
        f"merge sequence diverged at target={target_vocab}: "
        f"{len(inc.merge_rules)} inc vs {len(nai.merge_rules)} naive merges; "
        f"first diff at "
        f"{next((i for i, (a, b) in enumerate(zip(inc.merge_rules, nai.merge_rules)) if a != b), 'len')}"
    )

    # 2. Identical vocab (ids + fused vectors). merge_rules equal => vectors must match exactly.
    assert set(inc.vocab) == set(nai.vocab), "vocab id sets differ"
    for cid in inc.vocab:
        vi = inc.vocab[cid].vector
        vn = nai.vocab[cid].vector
        assert np.array_equal(vi, vn), f"fused vector mismatch at coord {cid} (target={target_vocab})"

    return len(inc.merge_rules), t_inc, t_naive


def _check(target_vocab: int) -> None:
    corpus = _corpus(8000)
    n_merges, _t_inc, _t_naive = _assert_equivalent(target_vocab, corpus)
    # sanity: the trainers actually did work (not a trivial 0-merge pass)
    assert n_merges > 0, f"no merges happened at target={target_vocab} (corpus too small / min_count too high)"


def _check_corpus(corpus: bytes, target_vocab: int) -> None:
    n_merges, _ti, _tn = _assert_equivalent(target_vocab, corpus)
    assert n_merges > 0, f"no merges happened (target={target_vocab}, {len(corpus)}B corpus)"


if pytest is not None:

    @pytest.mark.parametrize("target_vocab", [300, 512, 1024, 4096])
    def test_incremental_equals_naive(target_vocab: int) -> None:
        _check(target_vocab)

    @pytest.mark.parametrize("target_vocab", [300, 512, 1024])
    def test_incremental_equals_naive_unicode(target_vocab: int) -> None:
        _check_corpus(_unicode_corpus(8000), target_vocab)

    @pytest.mark.parametrize("target_vocab", [300, 400])
    def test_incremental_equals_naive_reverse_pairs(target_vocab: int) -> None:
        _check_corpus(_reverse_pair_corpus(6000), target_vocab)

    def test_select_best_pair_tiebreak_is_lexicographic() -> None:
        """Regression lock for the merge-121 divergence: on an exact score tie the
        naive oracle (`_select_best_pair`) MUST pick the lexicographically-smallest
        pair, regardless of dict-insertion order — matching the incremental path's
        rule. Pre-fix this returned the first-inserted pair (5,3) and the two
        trainers produced different vocabularies. Insertion order here is
        deliberately the OPPOSITE of lexicographic order so a regression is caught.
        """
        c = FisherCoordizer(target_vocab_size=300)
        # Exact tie: identical count/coupling/entropy. Insert (5,3) before (3,5),
        # and a non-reverse tie (9,1) before (2,2).
        tie = {"count": 10, "coupling": 50.0, "entropy": 0.0, "contexts": []}
        ps = {
            (5, 3): dict(tie),
            (3, 5): dict(tie),
            (9, 1): dict(tie),
            (2, 2): dict(tie),
        }
        pick = c._select_best_pair(ps)
        assert pick == (2, 2), (
            f"naive tie-break not canonical (lexicographic): picked {pick}, "
            f"expected (2, 2) — the merge-121 divergence has regressed"
        )

    def test_strict_max_still_wins_over_tiebreak() -> None:
        """A strictly higher score must beat a lexicographically-smaller pair —
        the tie-break only applies on exact ties, never overriding the max."""
        c = FisherCoordizer(target_vocab_size=300)
        ps = {
            (1, 1): {"count": 5, "coupling": 10.0, "entropy": 0.0, "contexts": []},   # score 500
            (9, 9): {"count": 10, "coupling": 50.0, "entropy": 0.0, "contexts": []},  # score 5000
        }
        assert c._select_best_pair(ps) == (9, 9)


if __name__ == "__main__":
    corpus = _corpus(8000)
    print(f"Equivalence gate: incremental == naive  (corpus={len(corpus):,} bytes)\n")
    print(f"{'target':>8} {'merges':>8} {'t_inc(s)':>10} {'t_naive(s)':>11} {'speedup':>8}  result")
    print("-" * 64)
    all_ok = True
    for target in (300, 512, 1024, 4096):
        try:
            n, ti, tn = _assert_equivalent(target, corpus)
            speed = (tn / ti) if ti > 0 else float("inf")
            print(f"{target:>8} {n:>8} {ti:>10.3f} {tn:>11.3f} {speed:>7.1f}x  ✅ PASS")
        except AssertionError as e:
            all_ok = False
            print(f"{target:>8} {'-':>8} {'-':>10} {'-':>11} {'-':>8}  ❌ FAIL")
            print(f"         {e}")
    print("-" * 64)
    print("ALL PASS ✅" if all_ok else "DIVERGENCE DETECTED ❌")
