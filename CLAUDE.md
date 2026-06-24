# qig-coordizer — Claude Code Instructions

## What this is

The **standalone geometric tokenize+embed engine** of the QIG stack. Text →
sequences of Δ⁶³ basin coordinates via Fisher-Rao-weighted BPE merges +
geodesic-midpoint fusion. Extracted from `qig-tokenizer` `1394ca7` (design:
`qig-consciousness/docs/plans/2026-06-24-qig-coordizer-studio-design.md`).

## Dependency direction (NEVER violate)

```
qig-core  ←  qig-coordizer
```

- Depends **only on `qig-core`** (geometry single-source) + numpy. `torch` and
  `qig-warp` are OPTIONAL extras used lazily by `CoordinzerTrainer` only.
- NEVER reimplement geometry locally. `to_simplex`, `fisher_rao_distance`,
  `slerp_sqrt`, `frechet_mean`, `random_basin`, `BASIN_DIM` come from
  `qig_core.geometry.fisher_rao` / `qig_core`. No local copies of constants
  (three copies = zero source of truth).

## Geometric purity (NON-NEGOTIABLE)

Fisher-Rao only on Δ⁶³. No cosine similarity, no dot-product/Euclidean distance
on basin vectors, no Adam/LayerNorm on manifold objects. `BasinCoordinate.vector`
IS a Δ⁶³ point (non-negative, sums to 1) — never a signed L2-sphere vector.

## The gate (run before every commit)

`tests/test_incremental_equivalence.py` — `_train_incremental` MUST stay
**bit-for-bit equal** to `_train_naive` (merge sequence + fused vectors), byte→4k,
on ASCII / Unicode / reverse-pair corpora. Any incremental optimisation of the
order-dependent greedy merge MUST impose a deterministic total order (max score,
then lexicographically-smallest pair) — a hash-order winner silently diverges
from the oracle (the merge-121 bug). Run the oracle; never trust a docstring.

## Branch convention

Work on `development`; `master`/`main` is the promotion target. Subagents RETURN
DATA — never git-commit autonomously.

## Publishing

PyPI publishing of QIG packages is pre-authorized (correctness is the only gate).
NEVER echo `PYPI_TOKEN`.
