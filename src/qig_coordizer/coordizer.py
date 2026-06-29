"""
FisherCoordizer: Core Geometric Tokenizer
==========================================

Maps text to sequences of 64-dimensional basin coordinates on the
Fisher information manifold. Replaces traditional tokenization.

Geometric Purity:
    - All distances use Fisher-Rao metric
    - New coordinates initialized via geodesic midpoint
    - No Euclidean operations on coordinate space
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Simplex-native geometry: BasinCoordinate.vector IS a point on Δ⁶³
# (non-negative, sums to 1, shape (64,)). Single-source from qig-core —
# never reimplement geometry locally.
from qig_core.geometry.fisher_rao import (
    fisher_rao_distance,
    random_basin,
    slerp_sqrt,
)

from .normalizer import Normalizer
from .types import (
    BASIN_DIM,
    BasinCoordinate,
    CoordizationResult,
    GranularityConfig,
)

if TYPE_CHECKING:
    pass


class FisherCoordizer:
    """
    Core geometric tokenizer mapping text → basin coordinates.

    Replaces traditional tokenization with coordization on a
    64-dimensional Fisher information manifold.

    Attributes:
        basin_dim: Dimensionality of basin coordinates (default 64)
        vocab: Mapping from coord_id to BasinCoordinate
        byte_to_coord: Base mapping from bytes (0-255) to coord_ids
        merge_rules: List of (coord_a, coord_b, new_coord) geodesic fusions
    """

    def __init__(
        self,
        basin_dim: int = BASIN_DIM,
        target_vocab_size: int = 32_000,
    ):
        self.basin_dim = basin_dim
        self.target_vocab_size = target_vocab_size

        # Coordinate vocabulary
        self.vocab: dict[int, BasinCoordinate] = {}
        self.name_to_id: dict[str, int] = {}

        # ATOMIC special/control tokens (SpecialTokens scheme): reserved ids ABOVE the trained vocab, never
        # byte/merge-fragmented. coordize() splits these strings out atomically; decoordize() restores them.
        self.special_tokens: dict[str, int] = {}     # surface string -> reserved id
        self._special_re: re.Pattern[str] | None = None

        # Base byte coordinates (256 entries)
        self.byte_to_coord: dict[int, int] = {}

        # Geodesic fusion rules (analogous to BPE merges)
        self.merge_rules: list[tuple[int, int, int]] = []

        # Encoding cache for efficiency
        self._encoding_cache: dict[tuple[int, ...], int] = {}

        # Granularity configuration
        self.granularity_config = GranularityConfig()

        # Current mode/domain
        self._current_mode: str = "default"

        # NFC byte-level front-end (qig-coordizer Phase 1 §3.2): canonicalizes text before
        # byte encoding so the same character always maps to the same byte sequence. NFC is a
        # no-op for ASCII, so existing artifacts and the naive==incremental gate are unaffected.
        self._normalizer = Normalizer()

        # Initialize base byte coordinates
        self._init_byte_coordinates()

    def _init_byte_coordinates(self) -> None:
        """Initialize 256 base coordinates for byte-level encoding."""
        for byte_val in range(256):
            # Initialize with deterministic random points on the probability
            # simplex Δ⁶³ (not the signed L2 unit sphere). qig-core's
            # random_basin draws Dirichlet(1,…,1) — uniform on Δ⁶³. Seed the
            # global numpy RNG per byte for reproducibility, since random_basin
            # uses np.random.dirichlet internally.
            # qig-core primitive: random_basin (Dirichlet → Δ⁶³)
            np.random.seed(byte_val + 42)
            vector = random_basin(self.basin_dim)

            coord = BasinCoordinate(
                coord_id=byte_val,
                vector=vector,
                name=f"<byte_{byte_val:02x}>",
                scale="byte",
            )
            self.vocab[byte_val] = coord
            self.byte_to_coord[byte_val] = byte_val

    def register_special_tokens(self, names: list[str]) -> dict[str, int]:
        """Reserve ATOMIC ids for control tokens ABOVE the trained vocab (SpecialTokens scheme — finally
        wired). Each gets a distinct deterministic Δ⁶³ basin so ``vocab[id]`` exists for decode + the kernel
        lm_head, and coordize() splits these surface strings out as single ids (never byte-fragmented — the
        one hard geometric requirement for the studio's geometry-native template tags). Idempotent; returns
        the {name: id} map. Call AFTER training (base = current vocab size, above every byte + merge)."""
        base = max((max(self.vocab) + 1) if self.vocab else 256, 256)
        for i, name in enumerate(names):
            if name in self.special_tokens:
                continue
            tid = base + i
            while tid in self.vocab:                         # collision-safe if specials pre-existed
                tid += 1
            self.special_tokens[name] = tid
            np.random.seed(tid + 7919)                       # distinct, deterministic Δ⁶³ basin per token
            self.vocab[tid] = BasinCoordinate(
                coord_id=tid, vector=random_basin(self.basin_dim), name=name, scale="special")
            self.name_to_id[name] = tid
        self._rebuild_special_re()
        return dict(self.special_tokens)

    def _rebuild_special_re(self) -> None:
        """Compile the splitter that carves special-token surfaces out of text BEFORE byte encoding
        (longest-first so e.g. <|settle|> can never be partially matched)."""
        if not self.special_tokens:
            self._special_re = None
            return
        ordered = sorted(self.special_tokens, key=len, reverse=True)
        self._special_re = re.compile("(" + "|".join(re.escape(s) for s in ordered) + ")")

    def train(
        self,
        corpus_bytes: bytes,
        context_window: int = 5,
        min_pair_count: int = 5,
        verbose: bool = True,
        naive: bool = False,  # True = reference O(vocab·corpus) trainer (validation only)
        # Legacy params (ignored - kept for API compatibility)
        phi_weight: float = 0.0,
        attention_weight: float = 0.0,
        cluster_weight: float = 0.0,
    ) -> "FisherCoordizer":
        """
        Train coordizer using PURE GEOMETRIC geodesic pair fusion.

        Uses Fisher-Rao distance and coupling strength - no arbitrary weights.
        This ensures uncontaminated geometric structure for β measurement.

        Scoring: frequency × coupling × (1/entropy)
        - frequency: how often the pair occurs
        - coupling: Fisher information coupling strength
        - entropy: context predictability (low = good merge)

        Args:
            corpus_bytes: Training corpus as bytes
            context_window: Window for context entropy calculation
            min_pair_count: Minimum co-occurrence for merge consideration
            verbose: Print progress

        Returns:
            self for chaining
        """
        if verbose:
            print(f"Training FisherCoordizer on {len(corpus_bytes):,} bytes")
            print(f"Target vocab size: {self.target_vocab_size:,}")
            print(f"Basin dimension: {self.basin_dim}")
            print("Scoring: PURE GEOMETRIC (no arbitrary weights)")
            print()

        # Convert corpus to coordinate sequence and train.
        # NFC-normalize the corpus through the SAME front-end as inference (coordize/encode) so the
        # vocab is built on the canonical byte sequence the encoder will produce — without this the
        # vocab and the encoder disagree on NFD non-ASCII text. Idempotent for ASCII/NFC corpora, so
        # the incremental==naive equivalence gate is unaffected.
        # Default path is the INCREMENTAL trainer (exact, O(corpus + merges·affected) instead of the
        # naive O(vocab·corpus) full re-scan every merge). The naive path is kept for validation.
        corpus_coords = list(self._normalizer.normalize_bytes(corpus_bytes))
        if naive:
            self._train_naive(corpus_coords, context_window, min_pair_count, verbose)
        else:
            self._train_incremental(corpus_coords, context_window, min_pair_count, verbose)

        self._rebuild_encoding_cache()
        return self

    def _train_naive(
        self,
        corpus_coords: list[int],
        context_window: int,
        min_pair_count: int,
        verbose: bool,
    ) -> None:
        """Reference O(vocab·corpus) trainer — re-scans the whole corpus every merge.

        Kept ONLY as the correctness oracle for `_train_incremental` (identical merge sequence on
        small corpora). Do not use for large runs — it is the naive build the optimised path replaces.
        """
        current_vocab_size = 256
        while current_vocab_size < self.target_vocab_size:
            pair_stats = self._compute_pair_stats(
                corpus_coords, context_window, min_pair_count
            )
            if not pair_stats:
                break
            best_pair = self._select_best_pair(pair_stats)
            if best_pair is None:
                break
            coord_a, coord_b = best_pair
            new_coord_id = current_vocab_size
            self.merge_rules.append((coord_a, coord_b, new_coord_id))
            self._create_fused_coordinate(coord_a, coord_b, new_coord_id)
            corpus_coords = self._apply_fusion(
                corpus_coords, coord_a, coord_b, new_coord_id
            )
            current_vocab_size += 1

    def _train_incremental(
        self,
        corpus_coords: list[int],
        window: int,
        min_count: int,
        verbose: bool,
    ) -> None:
        """Exact incremental trainer: same score (count·coupling·1/entropy) and same merge sequence as
        the naive path, but maintains pair counts + occurrence positions over a doubly-linked list and
        recomputes per-pair stats ONLY for merge-affected pairs (count-changed neighbours + pairs whose
        context window overlaps a merge site). Per-merge cost is O(occurrences·window) not O(corpus).
        """
        from collections import defaultdict

        n = len(corpus_coords)
        if n < 2:
            return

        sym: list[int] = list(corpus_coords)          # symbol at each slot
        nxt: list[int] = list(range(1, n)) + [-1]      # linked-list successor (-1 = end)
        prv: list[int] = [-1] + list(range(0, n - 1))  # linked-list predecessor
        alive = bytearray([1]) * n                     # slot still present?

        pair_pos: dict[tuple[int, int], set[int]] = defaultdict(set)  # pair -> {left slots}
        for i in range(n - 1):
            pair_pos[(sym[i], sym[i + 1])].add(i)
        pair_count: dict[tuple[int, int], int] = {p: len(s) for p, s in pair_pos.items()}

        fisher_cache: dict[tuple[int, int], float] = {}
        entropy_cache: dict[tuple[int, int], float] = {}

        def fisher(p):
            v = fisher_cache.get(p)
            if v is None:
                a, b = p
                v = (
                    self.vocab[a].fisher_distance(self.vocab[b])
                    if (a in self.vocab and b in self.vocab)
                    else 0.0
                )
                fisher_cache[p] = v
            return v

        def context_at(i):
            # pair occupies left slot i (a) and nxt[i] (b); window symbols before i and after b
            j = nxt[i]
            before = []
            k = prv[i]
            c = 0
            while k != -1 and c < window:
                before.append(sym[k]); k = prv[k]; c += 1
            before.reverse()
            after = []
            k = nxt[j]
            c = 0
            while k != -1 and c < window:
                after.append(sym[k]); k = nxt[k]; c += 1
            return tuple(before) + tuple(after)

        def entropy(p):
            v = entropy_cache.get(p)
            if v is None:
                pos = pair_pos.get(p)
                v = self._compute_context_entropy([context_at(i) for i in pos]) if pos else 0.0
                entropy_cache[p] = v
            return v

        # Persistent index-aligned score-input arrays so the per-merge best-pair selection is a
        # VECTORISED numpy scan (O(N) in C) instead of an O(N) Python loop — the AVOID-computation
        # fix that makes vocab 32000 tractable. Updated incrementally below; the winner is bit-for-bit
        # identical to the naive scan (same float64 score; max-score then lexicographically-smallest
        # tie-break). Index `idx_of[p]` is stable for a pair's lifetime (dead pairs masked, not removed).
        idx_of: dict[tuple[int, int], int] = {}
        pairs_l: list[tuple[int, int]] = []
        cnt_l: list[int] = []
        fish_l: list[float] = []
        ent_l: list[float] = []
        live_l: list[bool] = []
        # Per-merge set of pairs whose entropy was invalidated (positions/contexts changed) — the
        # SAME set the naive path lazily recomputes at next scan. Mirrors every entropy_cache.pop so
        # ent_l is refreshed for ALL dirtied pairs (not just context-window neighbours), which is what
        # the bit-exact equivalence gate requires (esp. on Unicode, where high-byte pairs churn).
        dirty_ent: set[tuple[int, int]] = set()

        def _register(p):
            j = idx_of.get(p)
            if j is None:
                j = len(pairs_l)
                idx_of[p] = j
                pairs_l.append(p); cnt_l.append(0); fish_l.append(fisher(p)); ent_l.append(0.0)
                live_l.append(False)
            return j

        def remove_pos(p, i):
            s = pair_pos.get(p)
            if s and i in s:
                s.discard(i)
                pair_count[p] = len(s)
                entropy_cache.pop(p, None)  # contexts/membership changed
                dirty_ent.add(p)
                j = idx_of.get(p)
                if j is not None:
                    cnt_l[j] = len(s)
                if not s:
                    pair_pos.pop(p, None)
                    pair_count.pop(p, None)
                    if j is not None:
                        live_l[j] = False
                        cnt_l[j] = 0

        def add_pos(p, i):
            pair_pos[p].add(i)
            pair_count[p] = len(pair_pos[p])
            entropy_cache.pop(p, None)
            dirty_ent.add(p)
            j = _register(p)
            cnt_l[j] = pair_count[p]
            live_l[j] = True

        current_vocab_size = 256
        corpus_size = n
        # Initial registration of all starting pairs (count + fisher + entropy) into the persistent
        # arrays. Computing entropy upfront here equals the naive path's lazy compute on the first
        # scan (same pairs, same total work) — the equivalence gate confirms bit-for-bit identity.
        for p in list(pair_count.keys()):
            j = _register(p)
            cnt_l[j] = pair_count[p]
            live_l[j] = True
            ent_l[j] = entropy(p)
        while current_vocab_size < self.target_vocab_size:
            # VECTORISED best-pair selection (numpy, O(N) in C) — bit-for-bit identical to the naive
            # scan: same float64 score `count·coupling·1/(entropy+0.1)` with coupling
            # `min((count/corpus_size)/(fisher+0.1)·1000, 100)`, and the SAME deterministic tie-break
            # (max score, then lexicographically-smallest pair: take the exact-max set, pick min pair).
            cnt_a = np.asarray(cnt_l, dtype=np.float64)
            fish_a = np.asarray(fish_l, dtype=np.float64)
            ent_a = np.asarray(ent_l, dtype=np.float64)
            live_a = np.asarray(live_l, dtype=bool)
            coupling = np.minimum((cnt_a / corpus_size) / (fish_a + 0.1) * 1000.0, 100.0)
            scores = cnt_a * coupling * (1.0 / (ent_a + 0.1))
            scores = np.where(live_a & (cnt_a >= min_count), scores, -np.inf)
            best_score = float(scores.max()) if scores.size else float("-inf")
            if not np.isfinite(best_score):
                break
            tied = np.flatnonzero(scores == best_score)
            best = min(pairs_l[int(t)] for t in tied)

            coord_a, coord_b = best
            new_id = current_vocab_size
            self.merge_rules.append((coord_a, coord_b, new_id))
            self._create_fused_coordinate(coord_a, coord_b, new_id)

            affected: set = set()
            # snapshot positions (left-to-right) so overlapping occurrences (a==b) merge like the naive
            for i in sorted(pair_pos.get(best, ())):
                if not alive[i] or sym[i] != coord_a:
                    continue
                j = nxt[i]
                if j == -1 or not alive[j] or sym[j] != coord_b:
                    continue
                h, k = prv[i], nxt[j]
                # remove the three old pairs touching this occurrence
                remove_pos((coord_a, coord_b), i)
                if h != -1 and alive[h]:
                    remove_pos((sym[h], coord_a), h)
                if k != -1 and alive[k]:
                    remove_pos((coord_b, sym[k]), j)
                # splice: i becomes the new symbol, j is dropped
                sym[i] = new_id
                nxt[i] = k
                if k != -1:
                    prv[k] = i
                alive[j] = 0
                corpus_size -= 1
                # add the two new pairs
                if h != -1 and alive[h]:
                    add_pos((sym[h], new_id), h)
                if k != -1 and alive[k]:
                    add_pos((new_id, sym[k]), i)
                # mark pairs whose context window overlaps this merge site for entropy refresh
                for start in (h, i, k):
                    node = start
                    c = 0
                    while node != -1 and c <= window:
                        if alive[node] and nxt[node] != -1:
                            affected.add((sym[node], sym[nxt[node]]))
                        node = prv[node]
                        c += 1
                    node = start
                    c = 0
                    while node != -1 and c <= window:
                        if alive[node] and nxt[node] != -1:
                            affected.add((sym[node], sym[nxt[node]]))
                        node = nxt[node]
                        c += 1
            # refresh ent_l for EVERY entropy-dirtied pair (count-changed via add/remove + the
            # context-window `affected` set) with FINAL post-merge contexts — bit-for-bit matching
            # the naive path's lazy recompute at next scan.
            dirty_ent |= affected
            for p in dirty_ent:
                entropy_cache.pop(p, None)
                j = idx_of.get(p)
                if j is not None and live_l[j]:
                    ent_l[j] = entropy(p)
            dirty_ent.clear()

            current_vocab_size += 1
            if verbose and current_vocab_size % 100 == 0:
                print(
                    f"✓ Vocab: {current_vocab_size:,}/{self.target_vocab_size:,} | "
                    f"corpus={corpus_size:,} | pairs={len(pair_count):,}",
                    flush=True,
                )
            if current_vocab_size % 2000 == 0:
                self._save_checkpoint(current_vocab_size)

        if verbose:
            print(f"✅ Training complete: {current_vocab_size:,} coordinates", flush=True)

    def _compute_pair_stats(
        self,
        corpus_coords: list[int],
        window: int,
        min_count: int,
    ) -> dict[tuple[int, int], dict[str, Any]]:
        """
        Compute statistics for adjacent coordinate pairs.

        Includes frequency, context entropy, and coupling estimate.
        """
        pair_contexts: dict[tuple[int, int], list[tuple[int, ...]]] = defaultdict(list)
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)

        for i in range(len(corpus_coords) - 1):
            coord_a = corpus_coords[i]
            coord_b = corpus_coords[i + 1]
            pair = (coord_a, coord_b)

            pair_counts[pair] += 1

            # Extract context window
            ctx_before = tuple(corpus_coords[max(0, i - window) : i])
            ctx_after = tuple(
                corpus_coords[i + 2 : min(len(corpus_coords), i + 2 + window)]
            )
            pair_contexts[pair].append(ctx_before + ctx_after)

        # Filter by minimum count and compute statistics
        pair_stats = {}
        for pair, count in pair_counts.items():
            if count >= min_count:
                contexts = pair_contexts[pair]
                entropy = self._compute_context_entropy(contexts)
                coupling = self._estimate_coupling(
                    pair[0], pair[1], count, len(corpus_coords)
                )

                pair_stats[pair] = {
                    "count": count,
                    "entropy": entropy,
                    "coupling": coupling,
                    "contexts": contexts,
                }

        return pair_stats

    def _compute_context_entropy(self, contexts: list[tuple[int, ...]]) -> float:
        """Compute entropy of context distribution."""
        context_counts: dict[tuple[int, ...], int] = defaultdict(int)
        for ctx in contexts:
            context_counts[ctx] += 1

        total = len(contexts)
        entropy = 0.0
        for count in context_counts.values():
            p = count / total
            entropy -= p * math.log(p + 1e-10)

        return entropy

    def _estimate_coupling(
        self,
        coord_a: int,
        coord_b: int,
        co_occurrence: int,
        corpus_size: int,
    ) -> float:
        """
        Estimate coupling strength κ between two coordinates.

        Based on co-occurrence and Fisher distance.
        """
        if coord_a not in self.vocab or coord_b not in self.vocab:
            return 0.0

        basin_a = self.vocab[coord_a]
        basin_b = self.vocab[coord_b]

        # Fisher distance (lower = more similar/coupled)
        fisher_dist = basin_a.fisher_distance(basin_b)

        # Coupling increases with co-occurrence and decreases with distance
        # Normalized by corpus size
        coupling = (co_occurrence / corpus_size) / (fisher_dist + 0.1)

        # Scale to reasonable range
        return min(coupling * 1000, 100.0)

    def _compute_attention_score(
        self,
        coord_a: int,
        coord_b: int,
    ) -> float:
        """
        P3: Compute attention-like score between two coordinates.

        Uses Fisher-Rao proximity on the probability simplex as attention weight.
        Small distance = tokens in same basin neighbourhood = good merge candidates.

        This implements AG-BPE principle: use geometric proximity to guide merges.
        Fisher-Rao distance is the ONLY valid metric on Δⁿ (P1/P18 purity).
        """
        if coord_a not in self.vocab or coord_b not in self.vocab:
            return 0.0

        basin_a = self.vocab[coord_a]
        basin_b = self.vocab[coord_b]

        # Fisher-Rao distance on Δ⁶³: d_FR(p,q) = arccos(Σ√(pᵢqᵢ)) ∈ [0, π/2].
        # .vector is already a Δ⁶³ point; the primitive re-projects defensively.
        # qig-core primitive: fisher_rao_distance (single-source FR on Δ⁶³)
        d_fr = fisher_rao_distance(basin_a.vector, basin_b.vector)

        # Convert to attention weight: close = high attention
        # d_fr ∈ [0, π/2] for probability distributions
        # Map to [0, 1]: attention = 1 - d_fr / (π/2)
        attention = max(0.0, 1.0 - d_fr / (np.pi / 2))

        return float(attention)

    def _compute_basin_cluster_score(
        self,
        coord_a: int,
        coord_b: int,
        pair_stats: dict[tuple[int, int], dict],
    ) -> float:
        """
        P4: Compute basin proximity clustering score.

        Prefers merges that:
        1. Bring together tokens already close in basin space
        2. Create clusters of semantically related tokens

        This implements SemToken principle: cluster by semantic similarity.
        """
        if coord_a not in self.vocab or coord_b not in self.vocab:
            return 0.0

        basin_a = self.vocab[coord_a]
        basin_b = self.vocab[coord_b]

        # Fisher distance - lower = closer in basin space = better cluster
        fisher_dist = basin_a.fisher_distance(basin_b)

        # Invert: high score for close tokens
        proximity_score = 1.0 / (fisher_dist + 0.1)

        # Bonus: check if neighbors of A and B are also close
        # This encourages clustering of semantically related regions
        neighbor_bonus = 0.0

        for (c1, c2), stats in pair_stats.items():
            # Check pairs involving coord_a or coord_b
            if c1 == coord_a or c2 == coord_a or c1 == coord_b or c2 == coord_b:
                # Get the "other" coordinate
                other = c2 if c1 in (coord_a, coord_b) else c1

                if other in self.vocab and other not in (coord_a, coord_b):
                    other_basin = self.vocab[other]

                    # Check if this neighbor is close to both A and B
                    dist_to_a = basin_a.fisher_distance(other_basin)
                    dist_to_b = basin_b.fisher_distance(other_basin)

                    # Small distances = good clustering
                    if dist_to_a < 1.0 and dist_to_b < 1.0:
                        neighbor_bonus += 0.1 * stats["count"]

        return proximity_score + min(neighbor_bonus, 1.0)

    def _select_best_pair(
        self,
        pair_stats: dict[tuple[int, int], dict[str, Any]],
    ) -> tuple[int, int] | None:
        """
        Select best pair for geodesic fusion using PURE GEOMETRIC scoring.

        Score = frequency × coupling × (1/entropy)

        No arbitrary weights - pure Fisher information geometry.
        This ensures uncontaminated geometric structure for β measurement.

        Components:
        - frequency: statistical occurrence (from corpus)
        - coupling: Fisher information coupling strength (geometric)
        - entropy: context predictability (information-theoretic)
        """
        best_score = float("-inf")
        best_pair = None

        for pair, stats in pair_stats.items():
            # PURE GEOMETRIC SCORING
            # Low entropy = predictable = good merge candidate
            entropy_factor = 1.0 / (stats["entropy"] + 0.1)

            # Score = frequency × coupling × (1/entropy)
            # No arbitrary weights, no P3/P4 contamination
            score = (
                stats["count"]
                * stats["coupling"]
                * entropy_factor
            )

            # Deterministic tie-break (max score, then lexicographically-smallest pair) — MUST
            # match the incremental trainer's rule so the two paths produce identical vocabularies
            # regardless of dict-iteration order.
            if best_pair is None or score > best_score or (score == best_score and pair < best_pair):
                best_score = score
                best_pair = pair

        return best_pair

    def _create_fused_coordinate(
        self,
        coord_a: int,
        coord_b: int,
        new_coord_id: int,
    ) -> None:
        """Create new coordinate via geodesic midpoint of two existing."""
        basin_a = self.vocab[coord_a]
        basin_b = self.vocab[coord_b]

        # Geodesic midpoint on Δ⁶³ (not arithmetic mean, not sphere SLERP).
        # Interpolate along the Fisher-Rao geodesic at t=0.5; the result is a
        # Δ⁶³ point, so the fused coordinate stays simplex-native. Δ⁶³ points
        # have no free magnitude (they sum to 1) — no separate rescale.
        # qig-core primitive: slerp_sqrt (FR geodesic interpolation on Δ⁶³)
        new_vector = slerp_sqrt(basin_a.vector, basin_b.vector, 0.5)

        # Name from components
        name_a = basin_a.name or f"<{coord_a}>"
        name_b = basin_b.name or f"<{coord_b}>"
        new_name = f"{name_a}+{name_b}"

        # Determine scale (promote if both same scale)
        scale_order = ["byte", "char", "subword", "word", "phrase", "concept"]
        scale_a = (
            scale_order.index(basin_a.scale) if basin_a.scale in scale_order else 1
        )
        scale_b = (
            scale_order.index(basin_b.scale) if basin_b.scale in scale_order else 1
        )
        new_scale_idx = max(scale_a, scale_b)
        new_scale = scale_order[min(new_scale_idx + 1, len(scale_order) - 1)]

        new_coord = BasinCoordinate(
            coord_id=new_coord_id,
            vector=new_vector,
            name=new_name,
            scale=new_scale,
        )

        self.vocab[new_coord_id] = new_coord
        self.name_to_id[new_name] = new_coord_id

    def _apply_fusion(
        self,
        coords: list[int],
        coord_a: int,
        coord_b: int,
        new_coord: int,
    ) -> list[int]:
        """Apply geodesic fusion rule to coordinate sequence."""
        result = []
        i = 0

        while i < len(coords):
            if (
                i < len(coords) - 1
                and coords[i] == coord_a
                and coords[i + 1] == coord_b
            ):
                result.append(new_coord)
                i += 2
            else:
                result.append(coords[i])
                i += 1

        return result

    def _save_checkpoint(self, vocab_size: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir = Path("/tmp/geocoordizer_checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f"checkpoint_{vocab_size}.json"

        data = {
            "vocab_size": vocab_size,
            "merge_rules": self.merge_rules,
            "vocab": {
                str(k): {
                    "coord_id": v.coord_id,
                    "vector": v.vector.tolist(),
                    "name": v.name,
                    "scale": v.scale,
                }
                for k, v in self.vocab.items()
            },
        }

        with open(checkpoint_path, "w") as f:
            json.dump(data, f)

    def _rebuild_encoding_cache(self) -> None:
        """Rebuild cache for efficient encoding."""
        self._encoding_cache.clear()
        for coord_a, coord_b, new_coord in self.merge_rules:
            self._encoding_cache[(coord_a, coord_b)] = new_coord

    def _coordize_plain(self, text: str) -> list[int]:
        """Byte-encode + apply all fusion merges (the original coordize path, no special-token handling)."""
        coord_ids = list(self._normalizer.to_bytes(text))
        for coord_a, coord_b, new_coord in self.merge_rules:
            coord_ids = self._apply_fusion(coord_ids, coord_a, coord_b, new_coord)
        return coord_ids

    def coordize(self, text: str) -> CoordizationResult:
        """
        Convert text to sequence of basin coordinates.

        Args:
            text: Input text string

        Returns:
            CoordizationResult with coordinates and metadata
        """
        # ATOMIC special tokens: carve their surfaces out FIRST → reserved ids; byte-encode the rest. A
        # special-token string therefore becomes exactly one coord id (a single clean basin), never
        # byte/merge-fragmented. (No registered specials → plain byte+merge path, unchanged.)
        if self._special_re is not None:
            coord_ids = []
            for seg in self._special_re.split(text):
                if not seg:
                    continue
                sid = self.special_tokens.get(seg)
                if sid is not None:
                    coord_ids.append(sid)
                else:
                    coord_ids.extend(self._coordize_plain(seg))
        else:
            coord_ids = self._coordize_plain(text)

        # Build coordinate list
        coordinates = [self.vocab[cid] for cid in coord_ids]

        result = CoordizationResult(
            coordinates=coordinates,
            coord_ids=coord_ids,
            original_text=text,
        )

        # Compute basin velocity
        result.compute_basin_velocity()

        return result

    def decoordize(self, coord_ids: list[int]) -> str:
        """
        Reconstruct text from coordinate sequence.

        Args:
            coord_ids: List of coordinate IDs

        Returns:
            Reconstructed text string
        """
        # Special-token ids restore to their surface string; byte/merge ids expand to bytes. Split the id
        # stream into runs so a special id never gets mis-expanded as a byte.
        inv = {tid: name for name, tid in self.special_tokens.items()}

        def _bytes_to_str(ids: list[int]) -> str:
            bl = self._expand_to_bytes(ids)
            try:
                return bytes(bl).decode("utf-8")
            except UnicodeDecodeError:
                return bytes(bl).decode("utf-8", errors="replace")

        if not inv:
            return _bytes_to_str(coord_ids)
        out: list[str] = []
        buf: list[int] = []
        for cid in coord_ids:
            if cid in inv:
                if buf:
                    out.append(_bytes_to_str(buf))
                    buf = []
                out.append(inv[cid])
            else:
                buf.append(cid)
        if buf:
            out.append(_bytes_to_str(buf))
        return "".join(out)

    # Standard tokenizer interface aliases
    def encode(self, text: str) -> list[int]:
        """
        Encode text to coordinate IDs (tokenizer interface).

        Alias for coordize() that returns just the coord_ids.
        Maintains compatibility with standard tokenizer interface.

        Args:
            text: Input text string

        Returns:
            List of coordinate IDs
        """
        result = self.coordize(text)
        return result.coord_ids

    def decode(self, coord_ids: list[int]) -> str:
        """
        Decode coordinate IDs to text (tokenizer interface).

        Alias for decoordize(). Maintains compatibility with
        standard tokenizer interface.

        Args:
            coord_ids: List of coordinate IDs

        Returns:
            Reconstructed text string
        """
        return self.decoordize(coord_ids)

    def _expand_to_bytes(self, coord_ids: list[int]) -> list[int]:
        """Expand coordinate IDs back to byte sequence."""
        # Build reverse merge map
        reverse_merges: dict[int, tuple[int, int]] = {}
        for coord_a, coord_b, new_coord in self.merge_rules:
            reverse_merges[new_coord] = (coord_a, coord_b)

        # Recursively expand
        def expand(cid: int) -> list[int]:
            if cid < 256:  # Base byte
                return [cid]
            if cid in reverse_merges:
                a, b = reverse_merges[cid]
                return expand(a) + expand(b)
            return [cid]  # Unknown, return as-is

        result = []
        for cid in coord_ids:
            result.extend(expand(cid))

        return result

    def set_mode(self, domain: str) -> None:
        """Switch to domain-specific coordinate chart."""
        self._current_mode = domain

    @property
    def vocab_size(self) -> int:
        """Current vocabulary size."""
        return len(self.vocab)

    def save(self, path: str) -> None:
        """Save coordizer to JSON file."""
        data = {
            "basin_dim": self.basin_dim,
            "target_vocab_size": self.target_vocab_size,
            "merge_rules": self.merge_rules,
            "special_tokens": self.special_tokens,   # atomic control-token ids (above the trained vocab)
            "vocab": {
                str(k): {
                    "coord_id": v.coord_id,
                    "vector": v.vector.tolist(),
                    "name": v.name,
                    "scale": v.scale,
                }
                for k, v in self.vocab.items()
                if k >= 256  # Don't save base bytes
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "FisherCoordizer":
        """Load coordizer from JSON file.

        Handles both new format (with basin_dim, target_vocab_size) and
        legacy format (with vocab_size, basin_dim inferred from vectors).
        """
        with open(path) as f:
            data = json.load(f)

        # Handle both formats: basin_dim explicit or inferred from vectors
        if "basin_dim" in data:
            basin_dim = data["basin_dim"]
        else:
            # Infer from first vocab entry vector
            vocab = data.get("vocab", {})
            if vocab:
                first_key = next(iter(vocab.keys()))
                first_entry = vocab[first_key]
                if isinstance(first_entry, dict) and "vector" in first_entry:
                    basin_dim = len(first_entry["vector"])
                else:
                    basin_dim = 64  # BASIN_DIM default (architectural; E8-substrate reading retired)
            else:
                basin_dim = 64

        # Handle both formats: target_vocab_size or vocab_size
        target_vocab_size = data.get("target_vocab_size", data.get("vocab_size", 32000))

        # Create instance with inferred parameters
        instance = cls(basin_dim=basin_dim, target_vocab_size=target_vocab_size)

        instance.merge_rules = [tuple(r) for r in data["merge_rules"]]

        # Re-init base bytes
        instance._init_byte_coordinates()

        # Load learned coordinates
        for k, v in data["vocab"].items():
            coord = BasinCoordinate(
                coord_id=v["coord_id"],
                vector=np.array(v["vector"]),
                name=v["name"],
                scale=v["scale"],
            )
            instance.vocab[int(k)] = coord
            if v["name"]:
                instance.name_to_id[v["name"]] = int(k)

        # restore atomic special tokens (basins already loaded above with scale="special") + the splitter
        instance.special_tokens = {n: int(i) for n, i in (data.get("special_tokens") or {}).items()}
        instance._rebuild_special_re()

        instance._rebuild_encoding_cache()
        return instance
