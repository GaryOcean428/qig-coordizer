"""Normalizer front-end for the byte-level coordizer (qig-coordizer Phase 1).

Design: docs/20260624-qig-coordizer-stack-design-1.00W.md §3.2.

The multibyte-garbage failure mode it addresses
------------------------------------------------
The coordizer is byte-level: text is UTF-8 encoded and each byte (0-255) seeds an
independent Δ⁶³ basin. Without a normalization front-end, the SAME visible character
can arrive as different byte sequences (Unicode composed `é` U+00E9 = 2 bytes vs
decomposed `e`+◌́ U+0065 U+0301 = 3 bytes), so identical-looking text trains/encodes
to different coordinate sequences. NFC canonicalizes first, so a character always maps
to one byte sequence.

What it does NOT do (by design): it does not turn a multi-byte character into a single
token — the byte-level basis is KEPT (the design's explicit choice; do not switch to
char-level). A 3-byte character is still 3 byte tokens; NFC only makes WHICH 3 bytes
deterministic. The optional whitespace/punctuation pre-tokenize boundary is the lever
that stops merges from crossing word/character ends (which is what produces the U+FFFD
"" replacement-char garbage on decode when a merge splits a character mid-sequence).

Stdlib-only (unicodedata + re) so the shipped `qig_coordizer` package stays
self-contained; `qig_coordizer.normalizer` re-exports this (consolidation, Decision A2).
"""

from __future__ import annotations

import re
import unicodedata

# GPT-2/byte-level-BPE-style pre-tokenization boundary. Keeps a leading space attached to
# the following word (the SentencePiece ``▁`` convention) and isolates runs of letters,
# digits, and punctuation so a downstream merge engine never fuses across these boundaries.
_PRETOKEN_RE = re.compile(
    r"""\s*\w+|\s*[^\w\s]+|\s+""",
    re.UNICODE,
)

_VALID_FORMS = ("NFC", "NFD", "NFKC", "NFKD")


class Normalizer:
    """NFC (default) Unicode normalizer + optional pre-tokenize boundary for byte-level coordization.

    Parameters
    ----------
    form:
        Unicode normalization form. Default ``"NFC"`` (canonical composition) — the right
        default for a byte-level model: it collapses composed/decomposed duplicates without
        the lossy compatibility folding of NFKC/NFKD.
    pretokenize:
        If ``True``, :meth:`to_byte_segments` splits text on word/whitespace/punctuation
        boundaries so the trainer can confine merges within a segment. Default ``False``
        (Phase-1 fix is NFC; the boundary is the optional refinement).
    """

    def __init__(self, form: str = "NFC", pretokenize: bool = False) -> None:
        if form not in _VALID_FORMS:
            raise ValueError(f"form must be one of {_VALID_FORMS}, got {form!r}")
        self.form = form
        self.pretokenize = pretokenize

    # -- text -> text ---------------------------------------------------------------
    def normalize_text(self, text: str) -> str:
        """Return ``text`` in the configured Unicode normal form (idempotent for NFC ASCII)."""
        return unicodedata.normalize(self.form, text)

    # -- text -> bytes (the chokepoint that replaces raw ``text.encode("utf-8")``) ---
    def to_bytes(self, text: str) -> bytes:
        """Canonical text->bytes: normalize, then UTF-8 encode. This is the single conversion
        both training and inference MUST use so the vocab and the encoder agree byte-for-byte."""
        return self.normalize_text(text).encode("utf-8")

    def normalize_bytes(self, data: bytes) -> bytes:
        """NFC-normalize a TRAINING CORPUS given as bytes (decode UTF-8 -> normalize -> re-encode).

        This is the train-side twin of :meth:`to_bytes`: it guarantees the vocab is built from
        the SAME canonical byte sequence that inference (``coordize``/``encode``) will produce,
        closing the train/inference asymmetry. Idempotent for already-NFC / ASCII corpora, so the
        incremental==naive equivalence gate is unaffected. Falls back to the raw bytes if they are
        not valid UTF-8 (an arbitrary-byte corpus has no text to canonicalize; the byte-level basis
        is still valid)."""
        try:
            return self.to_bytes(data.decode("utf-8"))
        except UnicodeDecodeError:
            return data

    # -- optional pre-tokenize boundary --------------------------------------------
    def pretokenize_text(self, text: str) -> list[str]:
        """Split normalized ``text`` into pre-token segments (word / whitespace / punctuation runs).

        The concatenation of the segments reconstructs the normalized text exactly (lossless),
        so it is safe as a boundary marker without changing the covered bytes.
        """
        norm = self.normalize_text(text)
        segments = _PRETOKEN_RE.findall(norm)
        return segments

    def to_byte_segments(self, text: str) -> list[list[int]]:
        """Pre-token boundary as a list of byte-id segments. A merge engine that never fuses
        ACROSS segments cannot create a merge that splits a character or crosses a word end."""
        if not self.pretokenize:
            return [list(self.to_bytes(text))]
        return [list(seg.encode("utf-8")) for seg in self.pretokenize_text(text)]
