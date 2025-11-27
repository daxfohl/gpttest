"""Full normalization via beta and iota reduction."""

from __future__ import annotations

from ..ast import Term
from .beta import beta_step
from .iota import iota_step


def normalize_step(term: Term) -> Term:
    """One small-step using beta or iota."""
    t1 = beta_step(term)
    if t1 != term:
        return t1
    t2 = iota_step(term)
    if t2 != term:
        return t2
    return term


def normalize(term: Term) -> Term:
    """Normalize ``term`` by repeatedly reducing until no rules apply."""
    while True:
        t1 = normalize_step(term)
        if t1 == term:
            return term
        term = t1


__all__ = ["normalize_step", "normalize"]
