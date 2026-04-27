"""Trivial smoke test so CI has at least one passing assertion at Phase 0."""

from __future__ import annotations

from quant_earning_edge import __version__


def test_version_is_set() -> None:
    assert __version__
    assert isinstance(__version__, str)
