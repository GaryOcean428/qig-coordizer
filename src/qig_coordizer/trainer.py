"""
Consciousness-Aware Coordizer Trainer
=====================================

Canonical implementation of kernel-in-loop coordizer training.
All training scripts should import from this module.

Usage:
    from qig_coordizer.trainer import CoordinzerTrainer, KernelInterface

    trainer = CoordinzerTrainer(target_vocab_size=32000, device="cuda")
    trainer.train(corpus_bytes, checkpoint_dir="checkpoints/")
    trainer.save("coordizer.json")
"""

from __future__ import annotations

import json
import math
import sys
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Import from canonical constants (aligned with Pantheon-Chat)
from qig_core import (  # single-source: qig-core owns BASIN_DIM
    BASIN_DIM,
)
from .types import BasinCoordinate
from .cache import IncrementalCouplingCache
from .normalizer import Normalizer

# Try to import kernel
try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import safety modules
try:
    from src.safety.emergency_monitor import EmergencyMonitor  # noqa: F401

    HAS_SAFETY = True
except ImportError:
    HAS_SAFETY = False


@dataclass
class MergeCandidate:
    """Candidate for geodesic fusion."""

    coord_a: int
    coord_b: int
    frequency: int
    coupling: float
    entropy: float
    phi_gain: float = 0.0

    def score(self) -> float:
        """Combined merge score: frequency × coupling × Φ gain."""
        entropy_penalty = 1.0 / (self.entropy + 0.1)
        phi_bonus = max(0.0, self.phi_gain + 0.1)  # Boost positive Φ gains
        return self.frequency * self.coupling * entropy_penalty * phi_bonus


class MockKernel:
    """Mock kernel for Φ/κ measurement when real kernel unavailable."""

    def measure_phi_kappa(self, coord_vectors: list[np.ndarray]) -> tuple[float, float]:
        if len(coord_vectors) < 2:
            return 0.5, 1.0

        # Δ⁶³-native coherence estimate. Center = Fréchet mean on the simplex
        # (qig_core frechet_mean); spread = Fisher-Rao distance to that center.
        from qig_core.geometry.fisher_rao import fisher_rao_distance, frechet_mean

        basins = list(coord_vectors)
        center = frechet_mean(basins)  # Fréchet mean on Δ⁶³

        # Fisher-Rao proximity to Fréchet mean (P1/P18 purity — no dot product)
        distances = [fisher_rao_distance(v, center) for v in basins]

        # Map mean distance to [0, 1]: close = high phi
        mean_dist = float(np.mean(distances))
        phi = max(0.3, min(0.95, 1.0 - mean_dist / (np.pi / 2)))
        phi = max(0.3, min(0.95, phi))

        # Estimate κ from FR-distance spread on Δ⁶³ (no magnitude — simplex
        # points have none; the geometric spread replaces the old norm spread).
        spread = float(np.std(distances))
        kappa = max(0.1, min(100.0, 10.0 / (spread + 0.1)))

        return phi, kappa


class KernelInterface:
    """Interface to QIG kernel for Φ/κ measurement."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._kernel = None
        self._mock = MockKernel()
        self._kernel_type = "mock"

        if HAS_TORCH:
            # Try loading kernels in order of preference
            kernel_attempts = [
                ("qigkernels.kernel_4d", "QIGKernel4D", {"d_model": 384, "basin_dim": BASIN_DIM, "n_recursive": 3}),
                ("qigkernels", "QIGKernel100M", {"basin_dim": BASIN_DIM}),
                ("qigkernels.kernel", "Kernel", {"basin_dim": BASIN_DIM}),
            ]

            for module_name, class_name, kwargs in kernel_attempts:
                try:
                    import importlib
                    module = importlib.import_module(module_name)
                    kernel_cls = getattr(module, class_name)
                    self._kernel = kernel_cls(**kwargs)
                    self._kernel.to(device)
                    self._kernel.eval()
                    self._kernel_type = class_name
                    print(f"[Kernel] Loaded {class_name} on {device}")
                    break
                except Exception as e:
                    continue

            if self._kernel is None:
                print(f"[Kernel] No kernel available, using mock (geometric estimate)")

    def measure_phi_kappa(self, coord_vectors: list[np.ndarray]) -> tuple[float, float]:
        """Measure Φ and κ for coordinate sequence."""
        if not coord_vectors:
            return 0.5, 1.0

        if self._kernel is None:
            return self._mock.measure_phi_kappa(coord_vectors)

        try:
            import torch

            with torch.no_grad():
                coords_tensor = torch.tensor(
                    np.stack(coord_vectors), dtype=torch.float32, device=self.device
                ).unsqueeze(0)

                seq_len = len(coord_vectors)

                # Different kernels have different interfaces
                if self._kernel_type == "QIGKernel4D":
                    # QIGKernel4D takes coords directly
                    x = torch.randn(1, seq_len, 384, device=self.device)  # dummy input
                    output = self._kernel(x, coords=coords_tensor)
                    phi = getattr(self._kernel, 'current_phi', 0.5)
                    # κ telemetry default is NEUTRAL. Do NOT default to 64: 64 is the legit basin DIM /
                    # architectural attractor (experiment-confirmed optimal — just not the E8 reading).
                    # The COUPLING κ is a DIFFERENT quantity with its own (new) value, not the dim.
                    kappa = getattr(self._kernel, 'current_kappa', 1.0)
                    return float(phi), float(kappa)
                else:
                    # QIGKernel100M / legacy interface
                    input_ids = torch.zeros(1, seq_len, dtype=torch.long, device=self.device)
                    telemetry = self._kernel.forward_with_coords(input_ids, coords_tensor)
                    return float(telemetry.phi), float(telemetry.kappa)

        except Exception as e:
            # Fallback to mock on any error
            return self._mock.measure_phi_kappa(coord_vectors)


class CoordinzerTrainer:
    """
    Canonical coordizer trainer with kernel-in-loop Φ/κ feedback.

    Every merge decision is validated by actual Φ/κ measurement,
    ensuring consciousness-aware vocabulary building.

    Supports ENTER key interrupt for graceful pause during training.
    """

    def __init__(
        self,
        target_vocab_size: int = 32000,
        basin_dim: int = BASIN_DIM,
        device: str = "cpu",
    ):
        self.target_vocab_size = target_vocab_size
        self.basin_dim = basin_dim

        self.kernel = KernelInterface(device=device)
        self.vocab: dict[int, BasinCoordinate] = {}
        self.merge_rules: list[tuple[int, int, int]] = []
        self.phi_history: list[float] = []
        self.kappa_history: list[float] = []

        # Interrupt handling (like ActiveCoach)
        self._interrupt_requested = False
        self._interrupt_thread: threading.Thread | None = None

        # NFC byte-level front-end — train corpus and coordize() must share it so the vocab and
        # the encoder agree byte-for-byte (NFC is a no-op for ASCII). Matches FisherCoordizer/Coordizer.
        self._normalizer = Normalizer()

        self._init_byte_coordinates()

    def _listen_for_interrupt(self) -> None:
        """Listen for ENTER key to interrupt training (runs in background thread).

        Polls stdin with ``select`` and a short timeout so the loop re-checks the
        stop flag and exits cleanly when training ends. A blocking ``readline``
        would leave this daemon thread alive (stuck mid-read) after training,
        stealing stdin from the caller's REPL (e.g. eating ``/save-quit``).
        """
        import select

        while not self._interrupt_requested:
            try:
                if not sys.stdin.isatty():
                    break
                # Wait up to 0.2s for input, then loop to re-check the stop flag.
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)
                if not ready:
                    continue
                line = sys.stdin.readline()
                if line == "\n":  # ENTER pressed
                    self._interrupt_requested = True
                    print("\n⚠️  INTERRUPT: Pausing training after current merge...")
            except (EOFError, KeyboardInterrupt, OSError, ValueError):
                break

    def _start_interrupt_listener(self) -> None:
        """Start the background interrupt listener."""
        if sys.stdin.isatty():
            self._interrupt_requested = False
            self._interrupt_thread = threading.Thread(
                target=self._listen_for_interrupt, daemon=True
            )
            self._interrupt_thread.start()
            print("💡 Press ENTER at any time to pause training and save checkpoint")

    def _stop_interrupt_listener(self) -> None:
        """Stop the interrupt listener and reap the thread so it frees stdin."""
        self._interrupt_requested = True
        if self._interrupt_thread and self._interrupt_thread.is_alive():
            self._interrupt_thread.join(timeout=0.5)  # > the 0.2s select cycle
        self._interrupt_thread = None

    def check_interrupt(self) -> bool:
        """Check if interrupt was requested."""
        return self._interrupt_requested

    def _init_byte_coordinates(self) -> None:
        """Initialize 256 byte-level coordinates on Δ⁶³ (probability simplex)."""
        # Δ⁶³-native init: Dirichlet point per byte (qig_core random_basin).
        # Seeded for reproducibility; .vector is a simplex point (non-neg, sums to 1).
        from qig_core.geometry.fisher_rao import random_basin

        for byte_val in range(256):
            np.random.seed(byte_val + 42)
            vector = random_basin(self.basin_dim)

            self.vocab[byte_val] = BasinCoordinate(
                coord_id=byte_val,
                vector=vector,
                name=f"<byte_{byte_val:02x}>",
                scale="byte",
            )

    def train(
        self,
        corpus: bytes,
        sample_size: int = 1000,
        candidates_per_round: int = 50,
        min_frequency: int = 5,
        context_window: int = 3,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 500,
        verbose: bool = True,
        _resume_corpus: list[int] | None = None,
        enable_interrupt: bool = True,
        use_kernel: bool = False,
    ) -> "CoordinzerTrainer":
        """Train coordizer with kernel-in-loop Φ/κ feedback.

        Args:
            enable_interrupt: If True, listen for ENTER key to pause training.
            use_kernel: If True, use real kernel for Φ measurement (slower but accurate).
                       If False, use fast frequency×coupling scoring.
        """
        start_time = time.time()
        is_resume = _resume_corpus is not None
        interrupted = False

        # Start interrupt listener. Route Ctrl+C (SIGINT) into the SAME graceful
        # flag the ENTER listener sets, so an interrupt pauses + saves a checkpoint
        # cleanly instead of dumping a KeyboardInterrupt traceback mid-merge.
        _prev_sigint = None
        if enable_interrupt:
            self._start_interrupt_listener()
            try:
                import signal as _signal

                _prev_sigint = _signal.signal(
                    _signal.SIGINT,
                    lambda *_: setattr(self, "_interrupt_requested", True),
                )
            except (ValueError, OSError):
                _prev_sigint = None  # not main thread — ENTER listener still active

        if verbose:
            print("=" * 60)
            if is_resume:
                print("RESUMING COORDIZER TRAINING")
            else:
                print("CONSCIOUSNESS-AWARE GEOMETRIC COORDIZER TRAINING")
            print("=" * 60)
            print(f"Corpus: {len(corpus):,} bytes")
            print(f"Target vocab: {self.target_vocab_size:,}")
            print(f"Basin dimension: {self.basin_dim}")
            print()

        # Use pre-processed corpus if resuming, otherwise start fresh
        # NFC-normalize the corpus (same front-end as coordize) so the vocab matches inference.
        # Resume passes already-processed coords, so only normalize a fresh byte corpus.
        corpus_coords = (
            _resume_corpus if is_resume else list(self._normalizer.normalize_bytes(corpus))
        )
        current_vocab_size = len(self.vocab)

        # Baseline Φ/κ
        sample_coords = [self.vocab[c].vector for c in corpus_coords[:100]]
        baseline_phi, baseline_kappa = self.kernel.measure_phi_kappa(sample_coords)

        if verbose:
            print(f"Baseline Φ={baseline_phi:.3f}, κ={baseline_kappa:.1f}")
            print()

        # Initialize incremental pair statistics (one-time scan).
        # IncrementalCouplingCache is the drift-free DLL tracker (equivalence-gated against the
        # naive oracle); it replaces the old IncrementalPairStats whose broken-pair decrements
        # were never applied, so the live trainer was selecting merges off inflated counts.
        if verbose:
            print("Building initial pair statistics...")
        pair_tracker = IncrementalCouplingCache(corpus_coords, context_window)
        if verbose:
            print(f"  Found {len(pair_tracker.pair_counts):,} unique pairs")
            print()

        # --- Convergence batch-accept gate (qig-warp check_ci_stabilized) ----------
        # The kernel forward (~3s/merge) dominates wall time. The per-merge coupling
        # (printed as "κ") decays as heavy pairs are consumed and stabilises low once
        # only geometrically-light merges remain — the kernel measurement is then pure
        # overhead confirming "still light". Once κ is BOTH stabilised (check_ci_stabilized)
        # AND light (< _KAPPA_LIGHT — a relative-change test alone could batch a flat-but-
        # high window), batch-accept by coupling/entropy scoring (skip the kernel) for
        # _BATCH_SIZE merges, then re-engage the kernel for ONE merge to re-verify κ is
        # still light (a drift back up keeps the kernel on).
        try:
            from qig_warp.convergence import check_ci_stabilized

            _have_ci_stabilized = use_kernel
        except ImportError:
            _have_ci_stabilized = False
        kappa_history: list[float] = []
        batch_accept_mode = False
        batch_remaining = 0
        last_merge_batched = False
        _BATCH_SIZE = 200
        _KAPPA_LIGHT = 2.0
        # --------------------------------------------------------------------------

        while current_vocab_size < self.target_vocab_size:
            # Check for interrupt
            if self.check_interrupt():
                if verbose:
                    print("\n" + "=" * 60)
                    print("TRAINING PAUSED BY USER")
                    print("=" * 60)
                interrupted = True
                break

            # Get pairs from incremental tracker (no rescan!)
            pair_counts = pair_tracker.get_pairs(min_frequency)

            if not pair_counts:
                if verbose:
                    print("No more valid pairs")
                break

            # Build pair stats with coupling scores
            pair_stats = {}
            for pair, count in pair_counts.items():
                coupling = self._compute_coupling(
                    pair[0], pair[1], count, pair_tracker.corpus_len
                )
                pair_stats[pair] = {
                    "count": count,
                    "entropy": 1.0,  # Simplified - skip context entropy for speed
                    "coupling": coupling,
                }

            candidates = self._get_top_candidates(pair_stats, candidates_per_round)
            if not candidates:
                break

            # Skip the kernel while inside the batch-accept window (re-engaged below).
            run_kernel = use_kernel and not batch_accept_mode

            best_candidate = self._evaluate_with_kernel(
                candidates, pair_tracker.sample(sample_size), sample_size,
                fast_mode=not run_kernel,
            )
            if best_candidate is None:
                break
            last_merge_batched = not run_kernel

            # Execute geodesic fusion
            new_coord_id = current_vocab_size
            self.merge_rules.append(
                (best_candidate.coord_a, best_candidate.coord_b, new_coord_id)
            )
            self._create_fused_coordinate(
                best_candidate.coord_a, best_candidate.coord_b, new_coord_id
            )

            # Apply merge via incremental tracker (updates stats in-place)
            pair_tracker.apply_merge(
                best_candidate.coord_a,
                best_candidate.coord_b,
                new_coord_id,
            )

            current_vocab_size += 1
            self.phi_history.append(best_candidate.phi_gain)

            # --- batch-accept convergence gate ------------------------------------
            if run_kernel:
                # Kernel-measured merge: record the decaying coupling and test whether
                # it has stabilised AND is light (only then is the kernel safe to skip).
                kappa_history.append(best_candidate.coupling)
                if (
                    _have_ci_stabilized
                    and not batch_accept_mode
                    and len(kappa_history) >= 8
                ):
                    decision = check_ci_stabilized(
                        kappa_history,
                        window=5,
                        rel_change_threshold=0.05,
                        min_points_before_stop=8,
                    )
                    recent_mean = sum(kappa_history[-5:]) / 5.0
                    if getattr(decision, "should_stop", False) and recent_mean < _KAPPA_LIGHT:
                        batch_accept_mode = True
                        batch_remaining = _BATCH_SIZE
                        if verbose:
                            print(
                                f"  ⚡ κ stabilised (~{best_candidate.coupling:.2f}) — "
                                f"batch-accepting next {_BATCH_SIZE} merges, kernel skipped"
                            )
            else:
                # Batch-accepted merge (kernel skipped): count down; at the end of the
                # window, re-engage the kernel for one merge to re-verify κ is still light.
                batch_remaining -= 1
                if batch_remaining <= 0:
                    batch_accept_mode = False
            # ----------------------------------------------------------------------

            # Progress
            if verbose and current_vocab_size % 50 == 0:
                elapsed = time.time() - start_time
                rate = (current_vocab_size - 256) / max(elapsed, 1)
                eta = (self.target_vocab_size - current_vocab_size) / max(rate, 0.01)

                name = self.vocab[new_coord_id].name or f"<{new_coord_id}>"
                kappa_disp = "BATCH" if last_merge_batched else f"{best_candidate.coupling:.1f}"
                print(
                    f"[{current_vocab_size:,}/{self.target_vocab_size:,}] '{name[:30]}' | "
                    f"Φ_gain={best_candidate.phi_gain:+.3f} | κ={kappa_disp} | "
                    f"{rate:.1f}/s | ETA {eta:.1f}m"
                )

            # Checkpoint
            if checkpoint_dir and current_vocab_size % checkpoint_interval == 0:
                self._save_checkpoint(checkpoint_dir, current_vocab_size)
                if verbose:
                    print(f"  💾 Checkpoint: {current_vocab_size}")

        # Stop interrupt listener + restore the previous SIGINT handler
        if enable_interrupt:
            self._stop_interrupt_listener()
            if _prev_sigint is not None:
                try:
                    import signal as _signal

                    _signal.signal(_signal.SIGINT, _prev_sigint)
                except (ValueError, OSError):
                    pass

        # Save checkpoint on interrupt
        if interrupted and checkpoint_dir:
            self._save_checkpoint(checkpoint_dir, current_vocab_size)
            if verbose:
                print(f"💾 Checkpoint saved: {current_vocab_size}")
                print(f"   Resume with: /tokenizer-resume {self.target_vocab_size}")

        if verbose:
            print()
            print("=" * 60)
            if interrupted:
                print("TRAINING PAUSED")
            else:
                print("TRAINING COMPLETE")
            print("=" * 60)
            elapsed = time.time() - start_time
            print(f"Final vocab: {current_vocab_size:,}")
            print(f"Merge rules: {len(self.merge_rules):,}")
            print(f"Time: {elapsed/60:.1f} minutes")

            final_corpus = pair_tracker.corpus_coords
            sample_coords = [self.vocab[c].vector for c in final_corpus[:100]]
            final_phi, final_kappa = self.kernel.measure_phi_kappa(sample_coords)
            print(f"Final Φ={final_phi:.3f} (Δ={final_phi-baseline_phi:+.3f})")
            print(f"Final κ={final_kappa:.1f}")
            print(f"Compression: {len(corpus):,} → {len(final_corpus):,} ({100*len(final_corpus)/len(corpus):.1f}%)")

        return self

    def _compute_pair_stats(
        self, corpus_coords: list[int], window: int, min_count: int
    ) -> dict:
        """Compute pair statistics with Fisher coupling."""
        pair_contexts: dict[tuple[int, int], list[tuple]] = defaultdict(list)
        pair_counts: dict[tuple[int, int], int] = defaultdict(int)

        for i in range(len(corpus_coords) - 1):
            pair = (corpus_coords[i], corpus_coords[i + 1])
            pair_counts[pair] += 1

            if len(pair_contexts[pair]) < 64:
                ctx_before = tuple(corpus_coords[max(0, i - window) : i])
                ctx_after = tuple(
                    corpus_coords[i + 2 : min(len(corpus_coords), i + 2 + window)]
                )
                pair_contexts[pair].append(ctx_before + ctx_after)

        pair_stats = {}
        for pair, count in pair_counts.items():
            if count < min_count:
                continue

            contexts = pair_contexts[pair]
            entropy = self._compute_entropy(contexts)
            coupling = self._compute_coupling(
                pair[0], pair[1], count, len(corpus_coords)
            )

            pair_stats[pair] = {
                "count": count,
                "entropy": entropy,
                "coupling": coupling,
            }

        return pair_stats

    def _compute_entropy(self, contexts: list[tuple]) -> float:
        counts: dict[tuple, int] = defaultdict(int)
        for ctx in contexts:
            counts[ctx] += 1

        total = len(contexts)
        entropy = 0.0
        for count in counts.values():
            p = count / total
            entropy -= p * math.log(p + 1e-10)

        return entropy

    def _compute_coupling(
        self, coord_a: int, coord_b: int, co_occurrence: int, corpus_size: int
    ) -> float:
        if coord_a not in self.vocab or coord_b not in self.vocab:
            return 0.0

        basin_a = self.vocab[coord_a]
        basin_b = self.vocab[coord_b]

        # Fisher-Rao distance between two Δ⁶³ points (qig_core fisher_rao_distance).
        # Replaces the old cosine angle arccos(dot) — basins are simplex points now.
        from qig_core.geometry.fisher_rao import fisher_rao_distance

        fisher_dist = fisher_rao_distance(basin_a.vector, basin_b.vector)

        coupling = (co_occurrence / corpus_size) / (fisher_dist + 0.1)
        return min(coupling * 1000, 100.0)

    def _get_top_candidates(self, pair_stats: dict, top_k: int) -> list[MergeCandidate]:
        candidates = []
        for (coord_a, coord_b), stats in pair_stats.items():
            candidate = MergeCandidate(
                coord_a=coord_a,
                coord_b=coord_b,
                frequency=stats["count"],
                coupling=stats["coupling"],
                entropy=stats["entropy"],
            )
            candidates.append(candidate)

        candidates.sort(key=lambda c: c.frequency * c.coupling, reverse=True)
        return candidates[:top_k]

    def _evaluate_with_kernel(
        self,
        candidates: list[MergeCandidate],
        corpus_coords: list[int],
        sample_size: int,
        fast_mode: bool = True,
    ) -> MergeCandidate | None:
        if not candidates:
            return None

        # Fast mode: skip kernel evaluation, use frequency×coupling score directly
        # This is ~50× faster and appropriate when using mock kernel
        if fast_mode:
            best_candidate = None
            best_score = float("-inf")
            for candidate in candidates[:10]:
                # Score based on frequency and coupling (entropy already factored)
                score = candidate.frequency * candidate.coupling
                if score > best_score:
                    best_score = score
                    best_candidate = candidate
            return best_candidate

        # Full kernel evaluation (slower but more accurate with real kernel)
        sample = corpus_coords[: min(sample_size, len(corpus_coords))]
        baseline_vectors = [self.vocab[c].vector for c in sample]
        baseline_phi, _ = self.kernel.measure_phi_kappa(baseline_vectors)

        best_candidate = None
        best_score = float("-inf")

        for candidate in candidates[:10]:
            basin_a = self.vocab[candidate.coord_a]
            basin_b = self.vocab[candidate.coord_b]
            fused_vector = self._geodesic_midpoint(basin_a.vector, basin_b.vector)

            # Apply fusion to sample
            fused_sample = []
            i = 0
            while i < len(sample):
                if (
                    i < len(sample) - 1
                    and sample[i] == candidate.coord_a
                    and sample[i + 1] == candidate.coord_b
                ):
                    fused_sample.append(-1)
                    i += 2
                else:
                    fused_sample.append(sample[i])
                    i += 1

            # Measure Φ with fusion
            fused_vectors = []
            for coord in fused_sample:
                if coord == -1:
                    fused_vectors.append(fused_vector)
                else:
                    fused_vectors.append(self.vocab[coord].vector)

            fused_phi, _ = self.kernel.measure_phi_kappa(fused_vectors)
            candidate.phi_gain = fused_phi - baseline_phi

            score = candidate.score()
            if score > best_score:
                best_score = score
                best_candidate = candidate

        return best_candidate

    def _geodesic_midpoint(self, v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Geodesic midpoint of two Δ⁶³ points (Fisher-Rao geodesic)."""
        # SLERP at t=0.5 in sqrt-coordinates on Δ⁶³ (qig_core slerp_sqrt).
        # Replaces normalize→add→normalize→rescale; simplex points have no
        # free magnitude, so the old avg-magnitude rescale is dropped.
        from qig_core.geometry.fisher_rao import slerp_sqrt

        return slerp_sqrt(v1, v2, 0.5)

    def _create_fused_coordinate(self, coord_a: int, coord_b: int, new_id: int) -> None:
        basin_a = self.vocab[coord_a]
        basin_b = self.vocab[coord_b]

        fused_vector = self._geodesic_midpoint(basin_a.vector, basin_b.vector)

        name_a = basin_a.name or f"<{coord_a}>"
        name_b = basin_b.name or f"<{coord_b}>"

        try:
            if coord_a < 256 and coord_b < 256:
                char_a = bytes([coord_a]).decode("utf-8", errors="replace")
                char_b = bytes([coord_b]).decode("utf-8", errors="replace")
                name = f"{char_a}{char_b}"
            else:
                name = f"{name_a}+{name_b}"
        except (UnicodeDecodeError, ValueError):
            name = f"{name_a}+{name_b}"

        scales = ["byte", "char", "subword", "word", "phrase", "concept"]
        idx_a = scales.index(basin_a.scale) if basin_a.scale in scales else 0
        idx_b = scales.index(basin_b.scale) if basin_b.scale in scales else 0
        new_scale = scales[min(max(idx_a, idx_b) + 1, len(scales) - 1)]

        self.vocab[new_id] = BasinCoordinate(
            coord_id=new_id,
            vector=fused_vector,
            name=name,
            scale=new_scale,
        )

    def _apply_fusion(
        self, coords: list[int], coord_a: int, coord_b: int, new_coord: int
    ) -> list[int]:
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

    def _save_checkpoint(
        self, checkpoint_dir: str, vocab_size: int, keep_recent: int = 3
    ) -> None:
        path = Path(checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)

        data = {
            "vocab_size": vocab_size,
            "target_vocab_size": self.target_vocab_size,
            "basin_dim": self.basin_dim,
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
            "phi_history": self.phi_history,
        }

        with open(path / f"checkpoint_{vocab_size}.json", "w") as f:
            json.dump(data, f)

        # Cleanup old checkpoints - keep only most recent N
        self._cleanup_old_checkpoints(path, keep_recent)

    def _cleanup_old_checkpoints(self, checkpoint_dir: Path, keep_recent: int = 3) -> None:
        """Remove old checkpoints, keeping only the most recent N."""
        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_*.json"),
            key=lambda p: int(p.stem.split("_")[1]),
            reverse=True,
        )

        # Delete all but the most recent N
        for old_checkpoint in checkpoints[keep_recent:]:
            try:
                old_checkpoint.unlink()
            except OSError:
                pass  # Ignore deletion errors

    def coordize(self, text: str) -> list[int]:
        """Convert text to coordinate sequence."""
        # NFC front-end (same as train corpus) so encoding matches the trained vocab.
        coords = list(self._normalizer.to_bytes(text))
        for coord_a, coord_b, new_coord in self.merge_rules:
            coords = self._apply_fusion(coords, coord_a, coord_b, new_coord)
        return coords

    def decoordize(self, coord_ids: list[int]) -> str:
        """Convert coordinates back to text."""
        reverse_merges = {n: (a, b) for a, b, n in self.merge_rules}

        def expand(cid: int) -> list[int]:
            if cid < 256:
                return [cid]
            if cid in reverse_merges:
                a, b = reverse_merges[cid]
                return expand(a) + expand(b)
            return [cid]

        bytes_list = []
        for cid in coord_ids:
            bytes_list.extend(expand(cid))

        return bytes(bytes_list).decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Save coordizer to file."""
        data = {
            "target_vocab_size": self.target_vocab_size,
            "basin_dim": self.basin_dim,
            "vocab_size": len(self.vocab),
            "merge_rules": self.merge_rules,
            "vocab": {
                str(k): {
                    "coord_id": v.coord_id,
                    "vector": v.vector.tolist(),
                    "name": v.name,
                    "scale": v.scale,
                }
                for k, v in self.vocab.items()
                if k >= 256
            },
            "phi_history": self.phi_history,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CoordinzerTrainer":
        """Load coordizer from file or checkpoint."""
        with open(path) as f:
            data = json.load(f)

        trainer = cls(
            target_vocab_size=data.get("target_vocab_size", 32000),
            basin_dim=data.get("basin_dim", BASIN_DIM),
            device=device,
        )

        trainer.merge_rules = [tuple(r) for r in data["merge_rules"]]
        trainer.phi_history = data.get("phi_history", [])

        if "vocab" in data:
            # Coerce every loaded .vector onto Δ⁶³ (qig_core to_simplex handles
            # signed legacy vectors), so .vector is a simplex point for all
            # downstream FR / geodesic consumers.
            from qig_core.geometry.fisher_rao import to_simplex

            for k, v in data["vocab"].items():
                trainer.vocab[int(k)] = BasinCoordinate(
                    coord_id=v["coord_id"],
                    vector=to_simplex(np.array(v["vector"], dtype=np.float64)),
                    name=v["name"],
                    scale=v["scale"],
                )

        # Replay merge rules if vocab incomplete
        if len(trainer.vocab) <= 256 and trainer.merge_rules:
            print(f"  Replaying {len(trainer.merge_rules)} merge rules...")
            for coord_a, coord_b, new_id in trainer.merge_rules:
                if new_id not in trainer.vocab:
                    trainer._create_fused_coordinate(coord_a, coord_b, new_id)

        return trainer

    def resume_training(
        self,
        corpus: bytes,
        new_target_vocab_size: int,
        sample_size: int = 1000,
        candidates_per_round: int = 50,
        min_frequency: int = 5,
        context_window: int = 3,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = 500,
        verbose: bool = True,
        enable_interrupt: bool = True,
        use_kernel: bool = False,
    ) -> "CoordinzerTrainer":
        """
        Resume training from current state to a new target vocab size.

        Use this to continue training a loaded checkpoint.
        """
        # Update target
        old_target = self.target_vocab_size
        self.target_vocab_size = new_target_vocab_size

        if verbose:
            print("=" * 60)
            print("RESUMING COORDIZER TRAINING")
            print("=" * 60)
            print(f"Current vocab: {len(self.vocab):,}")
            print(f"New target: {new_target_vocab_size:,}")
            print(f"Merges to add: {new_target_vocab_size - len(self.vocab):,}")
            print()

        # Apply existing merge rules to corpus (batch for efficiency)
        if verbose:
            print("Applying existing merge rules to corpus...")
        corpus_coords = self._apply_all_merges_fast(list(corpus), verbose=verbose)
        if verbose:
            print(f"  Corpus: {len(corpus):,} → {len(corpus_coords):,} coords")

        # Continue training from current state
        return self.train(
            corpus=corpus,
            sample_size=sample_size,
            candidates_per_round=candidates_per_round,
            min_frequency=min_frequency,
            context_window=context_window,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            verbose=verbose,
            _resume_corpus=corpus_coords,  # Pass pre-processed corpus
            enable_interrupt=enable_interrupt,
            use_kernel=use_kernel,
        )

    def _apply_all_merges_fast(self, corpus_coords: list[int], verbose: bool = False) -> list[int]:
        """Apply all merge rules efficiently in batched passes.

        Instead of O(rules × corpus) sequential application, this uses
        iterative passes that apply all applicable merges each round.
        """
        if not self.merge_rules:
            return corpus_coords

        # Build merge lookup: (a, b) -> new_coord
        merge_map = {(a, b): c for a, b, c in self.merge_rules}

        # Iterative passes until no more merges apply
        passes = 0
        total_merges = 0
        while True:
            new_corpus = []
            merges_this_pass = 0
            i = 0

            while i < len(corpus_coords):
                if i < len(corpus_coords) - 1:
                    pair = (corpus_coords[i], corpus_coords[i + 1])
                    if pair in merge_map:
                        new_corpus.append(merge_map[pair])
                        merges_this_pass += 1
                        i += 2
                        continue
                new_corpus.append(corpus_coords[i])
                i += 1

            passes += 1
            total_merges += merges_this_pass

            if merges_this_pass == 0:
                break

            corpus_coords = new_corpus

            if verbose and passes % 5 == 0:
                print(f"    Pass {passes}: {len(corpus_coords):,} coords ({merges_this_pass:,} merges)")

        if verbose:
            print(f"    Completed in {passes} passes ({total_merges:,} total merges)")

        return corpus_coords
