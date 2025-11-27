"""Weak head normal form reduction leveraging beta/iota head steps."""

from __future__ import annotations

from ..ast import Term
from .beta import beta_head_step
from .iota import iota_head_step


def whnf_step(term: Term) -> Term:
    """One small-step using beta or iota head reduction."""
    t1 = beta_head_step(term)
    if t1 != term:
        return t1
    t2 = iota_head_step(term)
    if t2 != term:
        return t2
    return term


def whnf(term: Term) -> Term:
    while True:
        t1 = whnf_step(term)
        if t1 == term:
            return term
        term = t1


__all__ = ["whnf_step", "whnf"]
