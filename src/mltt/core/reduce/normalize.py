"""Full normalization via beta and iota reduction."""

from __future__ import annotations

from ..ast import Term


def normalize_step(term: Term) -> Term:
    """One small-step using beta or iota."""

    return term.normalize_step()


def normalize(term: Term) -> Term:
    """Normalize ``term`` by repeatedly reducing until no rules apply."""

    return term.normalize()


__all__ = ["normalize_step", "normalize"]
