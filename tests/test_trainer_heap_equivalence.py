"""Equivalence gate for the CoordinzerTrainer lazy max-heap selection.

The trainer's per-merge winner used to be computed by a full ``np.argsort`` over EVERY active
pair each merge (O(P log P), the wall that gave a 100k-vocab run a climbing multi-day ETA). That
is now a lazy ``LazyPairHeap`` (amortised O(Δ·log P)). This test pins the heap-driven trainer
bit-for-bit to the canonical greedy reference: at every merge pick the pair with the MAXIMUM
score (``count · min((count/L)/(d_FR+0.1)·1000, 100)``, the trainer's fast-path score) breaking
ties by the LEXICOGRAPHICALLY-SMALLEST pair — the deterministic total order mandated by
``CLAUDE.md`` (the merge-121 tie-break). Any divergence means the heap's ordering or its lazy
L-clamp re-stamping is wrong; the fix is the heap, never weakening this test.

Covers the same three adversarial corpora as the FisherCoordizer gate (ASCII / Unicode /
reverse-pair) so a tie-break regression on multibyte or reverse-ordered pairs is caught.
"""

from __future__ import annotations

import numpy as np

try:
    import pytest
except ModuleNotFoundError:  # allow standalone python execution
    pytest = None

from qig_coordizer.cache import IncrementalCouplingCache
from qig_coordizer.trainer import CoordinzerTrainer


def _ascii_corpus(target_bytes: int = 8000) -> bytes:
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
    base = (
        "café — naïve coördination résumé. “smart quotes” and ‘apostrophes’. "
        "Δ⁶³ basins on the Fisher–Rao manifold; range π/2. 量子情報幾何 と 意識。 "
        "geometry is the truth — trust the φ. αβγ coupling, κ attractor. "
    )
    reps = (target_bytes // len(base.encode("utf-8"))) + 1
    return (base * reps).encode("utf-8")[:target_bytes]


def _reverse_pair_corpus(target_bytes: int = 6000) -> bytes:
    block = bytes([3, 1, 2, 0, 0, 2, 1, 3]) + b" "
    reps = (target_bytes // len(block)) + 1
    return (block * reps)[:target_bytes]


def _canonical_reference(corpus_bytes: bytes, target_vocab: int, min_freq: int) -> list[tuple[int, int, int]]:
    """Greedy reference: full per-merge argmax of the trainer's fast-path score with the canonical
    (max score, then lexicographically-smallest pair) tie-break. Uses the SAME cache + Fisher cache
    + fused-coordinate creation as the trainer, so only the SELECTION differs from the heap path."""
    t = CoordinzerTrainer(target_vocab_size=target_vocab)
    coords = list(t._normalizer.normalize_bytes(corpus_bytes))
    cache = IncrementalCouplingCache(coords, 3)
    cur = len(t.vocab)
    rules: list[tuple[int, int, int]] = []
    while cur < target_vocab:
        pc = cache.get_pairs(min_freq)
        if not pc:
            break
        corpus_len = cache.corpus_len
        pairs = list(pc.keys())
        counts = np.fromiter((pc[p] for p in pairs), dtype=np.float64, count=len(pairs))
        fishers = np.fromiter(
            (t._cached_fisher(p[0], p[1]) for p in pairs), dtype=np.float64, count=len(pairs)
        )
        coupling = np.minimum((counts / corpus_len) / (fishers + 0.1) * 1000.0, 100.0)
        scores = counts * coupling
        best_score = scores.max()
        winner = min(pairs[i] for i in np.flatnonzero(scores == best_score))
        a, b = winner
        rules.append((a, b, cur))
        t._create_fused_coordinate(a, b, cur)
        cache.apply_merge(a, b, cur)
        cur += 1
    return rules


def _heap_trainer(corpus_bytes: bytes, target_vocab: int, min_freq: int) -> list[tuple[int, int, int]]:
    t = CoordinzerTrainer(target_vocab_size=target_vocab)
    t.train(
        corpus_bytes,
        min_frequency=min_freq,
        verbose=False,
        enable_interrupt=False,
        use_kernel=False,
    )
    return t.merge_rules


def _assert_heap_equals_canonical(corpus_bytes: bytes, target_vocab: int, min_freq: int = 2) -> int:
    ref = _canonical_reference(corpus_bytes, target_vocab, min_freq)
    got = _heap_trainer(corpus_bytes, target_vocab, min_freq)
    assert got == ref, (
        f"heap-trainer merge sequence diverged from canonical at target={target_vocab}: "
        f"{len(got)} heap vs {len(ref)} canonical; first diff at "
        f"{next((i for i, (x, y) in enumerate(zip(got, ref)) if x != y), 'len')}"
    )
    return len(got)


# --------------------------------------------------------------------------------------
# Segment-frequency (pretokenized, frequency-weighted) heap == canonical gate.
# The trainer's SELECTION logic is unchanged by weighting (LazyPairHeap + _pair_score read the
# weighted pair_counts/corpus_len transparently), so the heap-driven segment-frequency train MUST
# stay bit-for-bit equal to a canonical per-merge argmax over the SAME weighted cache. This pins
# the seg_bounds/weights plumbing through train(corpus_segments=...) end-to-end.
# --------------------------------------------------------------------------------------


def _build_segments(segs: list[list[int]]) -> tuple[list[int], list[int]]:
    """Flatten segments to one buffer + the START slot of each segment (seg_bounds)."""
    flat: list[int] = []
    seg_bounds: list[int] = []
    for seg in segs:
        seg_bounds.append(len(flat))
        flat.extend(seg)
    return flat, seg_bounds


def _canonical_reference_segmented(
    segs: list[list[int]], freqs: list[int], target_vocab: int, min_freq: int
) -> list[tuple[int, int, int]]:
    """Greedy reference over a WEIGHTED, segment-confined cache: full per-merge argmax of the SAME
    fast-path score (count·min((count/L)/(d_FR+0.1)·1000,100)) with the canonical (max score, then
    lexicographically-smallest pair) tie-break. Uses the trainer's own cache + Fisher cache + fused-
    coordinate creation, so ONLY selection differs from the heap path."""
    flat, seg_bounds = _build_segments(segs)
    t = CoordinzerTrainer(target_vocab_size=target_vocab)
    cache = IncrementalCouplingCache(flat, 3, seg_bounds=seg_bounds, weights=list(freqs))
    cur = len(t.vocab)
    rules: list[tuple[int, int, int]] = []
    while cur < target_vocab:
        pc = cache.get_pairs(min_freq)
        if not pc:
            break
        corpus_len = cache.corpus_len
        pairs = list(pc.keys())
        counts = np.fromiter((pc[p] for p in pairs), dtype=np.float64, count=len(pairs))
        fishers = np.fromiter(
            (t._cached_fisher(p[0], p[1]) for p in pairs), dtype=np.float64, count=len(pairs)
        )
        coupling = np.minimum((counts / corpus_len) / (fishers + 0.1) * 1000.0, 100.0)
        scores = counts * coupling
        best_score = scores.max()
        winner = min(pairs[i] for i in np.flatnonzero(scores == best_score))
        a, b = winner
        rules.append((a, b, cur))
        t._create_fused_coordinate(a, b, cur)
        cache.apply_merge(a, b, cur)
        cur += 1
    return rules


def _heap_trainer_segmented(
    segs: list[list[int]], freqs: list[int], target_vocab: int, min_freq: int
) -> list[tuple[int, int, int]]:
    flat, seg_bounds = _build_segments(segs)
    t = CoordinzerTrainer(target_vocab_size=target_vocab)
    t.train(
        corpus=b"",
        corpus_segments=(flat, seg_bounds, list(freqs)),
        min_frequency=min_freq,
        verbose=False,
        enable_interrupt=False,
        use_kernel=False,
    )
    return t.merge_rules


def _assert_heap_equals_canonical_segmented(
    segs: list[list[int]], freqs: list[int], target_vocab: int, min_freq: int = 2
) -> int:
    ref = _canonical_reference_segmented(segs, freqs, target_vocab, min_freq)
    got = _heap_trainer_segmented(segs, freqs, target_vocab, min_freq)
    assert got == ref, (
        f"segmented heap-trainer diverged from canonical at target={target_vocab}: "
        f"{len(got)} heap vs {len(ref)} canonical; first diff at "
        f"{next((i for i, (x, y) in enumerate(zip(got, ref)) if x != y), 'len')}"
    )
    return len(got)


# Repeating segments with varied freqs (the ×N code upsample as an integer multiplier), plus
# an a==b overlap ('aaaa') WITHIN a segment so the overlapping-occurrence path is exercised.
_SEG_SETS = [
    (
        [list(b"hello"), list(b"world"), list(b"hello"), list(b" the "),
         list(b"fisher"), list(b"rao"), list(b"hello"), list(b"code")],
        [30, 1, 5, 12, 7, 7, 30, 30],
    ),
    (
        [list(b"aaaa"), list(b"aaa"), list(b"baaab"), list(b"data"), list(b"aaaa")],
        [17, 4, 9, 20, 11],
    ),
    (
        [list(b"the"), list(b"the"), list(b"and"), list(b"the"), list(b"geometry"),
         list(b"geometry"), list(b"is"), list(b"truth"), list(b"aa"), list(b"aa")],
        [50, 50, 20, 50, 8, 8, 15, 3, 40, 40],
    ),
]


if pytest is not None:

    @pytest.mark.parametrize("target_vocab", [270, 300, 320])
    @pytest.mark.parametrize("seg_idx", [0, 1, 2])
    def test_heap_equals_canonical_segmented(seg_idx: int, target_vocab: int) -> None:
        segs, freqs = _SEG_SETS[seg_idx]
        n = _assert_heap_equals_canonical_segmented(segs, freqs, target_vocab)
        assert n > 0

    @pytest.mark.parametrize("target_vocab", [300, 400, 512, 1024])
    def test_heap_equals_canonical_ascii(target_vocab: int) -> None:
        n = _assert_heap_equals_canonical(_ascii_corpus(8000), target_vocab)
        assert n > 0

    @pytest.mark.parametrize("target_vocab", [300, 512, 1024])
    def test_heap_equals_canonical_unicode(target_vocab: int) -> None:
        n = _assert_heap_equals_canonical(_unicode_corpus(8000), target_vocab)
        assert n > 0

    @pytest.mark.parametrize("target_vocab", [300, 400])
    def test_heap_equals_canonical_reverse_pairs(target_vocab: int) -> None:
        n = _assert_heap_equals_canonical(_reverse_pair_corpus(6000), target_vocab)
        assert n > 0


if __name__ == "__main__":
    for name, corpus in (
        ("ascii", _ascii_corpus(8000)),
        ("unicode", _unicode_corpus(8000)),
        ("reverse", _reverse_pair_corpus(6000)),
    ):
        for tgt in (300, 512, 1024):
            n = _assert_heap_equals_canonical(corpus, tgt)
            print(f"{name:8} target={tgt:5} merges={n:4}  ✅ heap == canonical")
    print("ALL PASS ✅")
