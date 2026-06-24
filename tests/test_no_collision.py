"""Guard against the qig_coordizer package-name collision (council red-team P0-F1).

The standalone qig-coordizer and qig-tokenizer's (now-retired) `src/qig_coordizer/`
staging re-export both exposed a top-level ``qig_coordizer``. In an editable env
with both on the path, ``import qig_coordizer`` silently resolved to whichever won
sys.path order — and the two had DIFFERENT surfaces (the tokenizer copy shipped a
boundary surface incl. ``InboundPath``; the standalone is the engine only).

This asserts the importable ``qig_coordizer`` is the STANDALONE engine, so a stray
co-install can't shadow it unnoticed.
"""

from __future__ import annotations

import inspect

import qig_coordizer


def test_qig_coordizer_is_the_standalone_engine():
    # Standalone exposes the engine surface...
    assert hasattr(qig_coordizer, "FisherCoordizer"), "standalone must expose FisherCoordizer"
    assert hasattr(qig_coordizer, "Normalizer")
    assert hasattr(qig_coordizer, "IncrementalCouplingCache")
    # ...and NOT the retired tokenizer-staging boundary surface. If InboundPath is
    # present, the tokenizer-staging copy won the name → collision regression.
    assert not hasattr(qig_coordizer, "InboundPath"), (
        "InboundPath present on qig_coordizer → the retired tokenizer-staging copy "
        "shadowed the standalone (name-collision regression, P0-F1)"
    )


def test_fishercoordizer_resolves_to_standalone_module():
    from qig_coordizer import FisherCoordizer

    assert FisherCoordizer.__module__ == "qig_coordizer.coordizer"
    # Stronger than __module__: the file must live in the standalone repo, not a copy.
    assert "qig_coordizer/coordizer.py" in inspect.getfile(FisherCoordizer)
