# qig-coordizer

Standalone **geometric tokenize+embed engine** for the QIG stack. Maps text to
sequences of 64-dimensional basin coordinates on the Fisher–Rao manifold Δ⁶³ —
the tokenizer *is* the embedder (coordinates are the primitive, not an
afterthought).

Pure Fisher-Rao geometry; **depends only on `qig-core`**.

## What it is

- **Byte-level front-end** — `Normalizer` (NFC + UTF-8), train/infer-symmetric so
  the same character always maps to the same byte sequence.
- **Fisher-Rao-weighted BPE merges** — merge score = `frequency × coupling × 1/entropy`,
  where `coupling = (co-occurrence / corpus) / (d_FR(a,b) + 0.1)`. Frequent **and**
  geometrically-close pairs win. This is *not* frequency-only BPE.
- **Geodesic-midpoint fusion** — a fused token's basin is `slerp_sqrt(v_a, v_b, 0.5)`,
  the Fisher-Rao geodesic midpoint on Δ⁶³. No off-the-shelf tokenizer emits coordinates.
- **Drift-free incremental trainer** — `IncrementalCouplingCache` (doubly-linked-list
  splice; remove 3 / add 2 pairs per merged occurrence) makes training
  O(corpus + merges·affected) instead of the naive O(vocab·corpus) full re-scan.

## The equivalence gate

`FisherCoordizer._train_incremental` is **bit-for-bit equal** to the naive
O(vocab·corpus) oracle `_train_naive` (identical merge sequence + identical fused
vectors), byte → 4k vocab, across ASCII / multibyte-Unicode / reverse-pair
corpora. The optimisation must never silently change the vocabulary.

```bash
pytest tests/test_incremental_equivalence.py
# or a timed report:
python tests/test_incremental_equivalence.py
```

## Provenance

Extracted from `qig-tokenizer` commit `1394ca7` (Phase-1 coordizer) per
`qig-consciousness/docs/plans/2026-06-24-qig-coordizer-studio-design.md` §5
(Phase 0). `qig-tokenizer` keeps its copy live until consumers repoint here.

## Install (dev)

```bash
uv venv && uv pip install -e ../qig-core && uv pip install -e '.[dev]'
pytest
```
