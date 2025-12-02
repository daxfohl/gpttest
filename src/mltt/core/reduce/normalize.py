"""Full normalization via beta and iota reduction."""

from __future__ import annotations

from .whnf import reduce_inside_step, whnf
from ..ast import Term


def normalize_step(term: Term) -> Term:
    """One small-step using beta or iota."""
    return reduce_inside_step(term, whnf)


def normalize(term: Term) -> Term:
    """Normalize ``term`` by repeatedly reducing until no rules apply."""
    while True:
        t1 = normalize_step(term)
        if t1 == term:
            return term
        term = t1


__all__ = ["normalize_step", "normalize"]
