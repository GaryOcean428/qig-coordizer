"""Unit tests for the coordizer NFC byte-level Normalizer front-end (qig-coordizer Phase 1 §3.2)."""

from __future__ import annotations

from qig_coordizer.normalizer import Normalizer


def test_ascii_is_noop():
    n = Normalizer()
    assert n.normalize_text("hello world") == "hello world"
    assert n.to_bytes("hello world") == b"hello world"


def test_nfc_collapses_composed_and_decomposed_to_same_bytes():
    """THE multibyte-garbage fix: the same visible character, whether typed pre-composed
    (U+00E9 'é') or decomposed ('e' U+0065 + combining acute U+0301), must produce ONE
    canonical byte sequence — otherwise it trains/encodes to two different coordinate paths."""
    n = Normalizer()
    composed = "é"          # é  (1 code point)
    decomposed = "é"       # e + combining acute (2 code points)

    # Without normalization these differ (2 bytes vs 3 bytes) — the bug.
    assert composed.encode("utf-8") != decomposed.encode("utf-8")

    # With NFC they are identical.
    assert n.to_bytes(composed) == n.to_bytes(decomposed) == b"\xc3\xa9"


def test_normalize_is_idempotent():
    n = Normalizer()
    for s in ("café", "é", "naïve", "Ω≈ω", "日本語", "ạ́"):
        once = n.normalize_text(s)
        assert n.normalize_text(once) == once


def test_byte_level_basis_kept_multibyte_is_multiple_bytes():
    """The design KEEPS the byte-level basis — NFC does not collapse a multi-byte char into
    one token; it only makes WHICH bytes deterministic. '日' is 3 UTF-8 bytes, still 3."""
    n = Normalizer()
    assert n.to_bytes("日") == "日".encode("utf-8")
    assert len(n.to_bytes("日")) == 3


def test_pretokenize_is_lossless():
    """Pre-token segments must concatenate back to the normalized text exactly (so using them
    as merge boundaries never drops or alters covered bytes)."""
    n = Normalizer(pretokenize=True)
    text = "the geometry is the truth, trust the Φ."
    segments = n.pretokenize_text(text)
    assert "".join(segments) == n.normalize_text(text)


def test_byte_segments_reconstruct_full_byte_stream():
    n_seg = Normalizer(pretokenize=True)
    n_flat = Normalizer(pretokenize=False)
    text = "hello   world  foo"
    seg_bytes = [b for seg in n_seg.to_byte_segments(text) for b in seg]
    assert seg_bytes == list(n_flat.to_bytes(text))
    # and pretokenize actually produced a boundary (more than one segment)
    assert len(n_seg.to_byte_segments(text)) > 1


def test_invalid_form_rejected():
    import pytest

    with pytest.raises(ValueError):
        Normalizer(form="NOPE")


def test_normalize_bytes_idempotent_and_fallback():
    """normalize_bytes (train-side twin of to_bytes): NFC-canonicalizes valid UTF-8, and
    falls back to raw bytes for non-UTF-8 input (byte-level basis still valid)."""
    import unicodedata

    n = Normalizer()
    nfd = unicodedata.normalize("NFD", "café").encode("utf-8")   # 5 bytes
    nfc = "café".encode("utf-8")                                  # 4 bytes (composed)
    assert nfd != nfc
    assert n.normalize_bytes(nfd) == n.normalize_bytes(nfc) == nfc
    # non-UTF-8 bytes pass through unchanged (no crash)
    raw = bytes([0xff, 0xfe, 0x00, 0x41])
    assert n.normalize_bytes(raw) == raw


def test_cap_splits_over_cap_ascii_into_bounded_chunks_lossless():
    """Char-safe cap: an over-cap ASCII run (e.g. the 1873-byte word concatenation / |----
    table-separator run the pre-flight found) splits into chunks each <= cap, and the
    concatenation reconstructs the original byte stream exactly (lossless)."""
    cap = 16
    n = Normalizer(max_segment_bytes=cap)
    text = "|" + "-" * 200 + "|"  # 202-byte no-whitespace ASCII run (>> cap)
    chunks = n.to_byte_segments(text)

    assert len(chunks) > 1, "an over-cap segment must actually be split"
    for chunk in chunks:
        assert len(chunk) <= cap, f"chunk of {len(chunk)} bytes exceeds cap {cap}"
    # lossless reconstruction of the full byte stream
    flat = [b for chunk in chunks for b in chunk]
    assert flat == list(n.to_bytes(text))
    assert bytes(flat).decode("utf-8") == n.normalize_text(text)


def test_cap_never_splits_mid_multibyte_character():
    """A cap that falls mid-multibyte-char must cut only at char boundaries (never mid-codepoint,
    the U+FFFD garbage this module exists to prevent). Every chunk must UTF-8-decode on its own,
    AND the concatenation must round-trip — for both CJK (3-byte) and Latin-accented (2-byte) runs."""
    for text, cap in (("量子情報幾何" * 5, 7), ("café" * 30, 3), ("量子" * 40, 2)):
        n = Normalizer(max_segment_bytes=cap)
        chunks = n.to_byte_segments(text)
        assert len(chunks) > 1, "an over-cap multibyte segment must be split"
        for chunk in chunks:
            # independent clean decode == the cut landed on a character boundary
            bytes(chunk).decode("utf-8")  # raises UnicodeDecodeError if it split mid-codepoint
        flat = [b for chunk in chunks for b in chunk]
        assert bytes(flat).decode("utf-8") == n.normalize_text(text)  # lossless round-trip


def test_cap_none_is_byte_identical_reduction():
    """Reduction proof: max_segment_bytes=None yields output byte-identical to no cap at all,
    in BOTH pretokenize modes — so every existing (uncapped) test stays green unchanged."""
    samples = ("hello   world  foo", "café résumé naïve", "|" + "-" * 200, "量子情報幾何", "")
    for pretok in (False, True):
        capped_none = Normalizer(pretokenize=pretok, max_segment_bytes=None)
        baseline = Normalizer(pretokenize=pretok)  # the pre-cap constructor signature
        for text in samples:
            assert capped_none.to_byte_segments(text) == baseline.to_byte_segments(text)


def test_train_infer_nfc_symmetry():
    """Regression for the council CRITICAL: FisherCoordizer.train() must NFC-normalize the corpus
    so the vocab matches NFC inference (coordize). Pre-fix, training on NFD bytes built a vocab
    the NFC encoder could never match. Training on decomposed (NFD) vs composed (NFC) text of the
    SAME content must now yield an identical vocab, and coordize(NFC) == coordize(NFD)."""
    import unicodedata

    from qig_coordizer import FisherCoordizer

    nfc_text = "café résumé naïve coördination " * 80
    nfd_text = unicodedata.normalize("NFD", nfc_text)
    assert nfc_text.encode("utf-8") != nfd_text.encode("utf-8")  # genuinely different byte streams

    c_nfc = FisherCoordizer(target_vocab_size=320)
    c_nfd = FisherCoordizer(target_vocab_size=320)
    c_nfc.train(nfc_text.encode("utf-8"), context_window=5, min_pair_count=2, verbose=False)
    c_nfd.train(nfd_text.encode("utf-8"), context_window=5, min_pair_count=2, verbose=False)

    # both corpora canonicalize to NFC internally -> identical vocab
    assert c_nfc.merge_rules == c_nfd.merge_rules, "train() not NFC-symmetric (the council CRITICAL)"

    # inference symmetric: composed and decomposed encode identically
    sample_nfc = "café résumé"
    sample_nfd = unicodedata.normalize("NFD", sample_nfc)
    assert c_nfc.coordize(sample_nfc).coord_ids == c_nfc.coordize(sample_nfd).coord_ids


if __name__ == "__main__":
    test_ascii_is_noop()
    test_nfc_collapses_composed_and_decomposed_to_same_bytes()
    test_normalize_is_idempotent()
    test_byte_level_basis_kept_multibyte_is_multiple_bytes()
    test_pretokenize_is_lossless()
    test_byte_segments_reconstruct_full_byte_stream()
    test_normalize_bytes_idempotent_and_fallback()
    test_cap_splits_over_cap_ascii_into_bounded_chunks_lossless()
    test_cap_never_splits_mid_multibyte_character()
    test_cap_none_is_byte_identical_reduction()
    test_train_infer_nfc_symmetry()
    print("normalizer: all standalone checks passed ✅")
