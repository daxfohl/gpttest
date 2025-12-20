"""Weak head normal form reduction leveraging beta/iota head steps."""

from __future__ import annotations

from typing import Callable

from ..ast import Term, Reducer


def whnf(term: Term) -> Term:
    """Wrapper around ``Term.whnf`` for backwards compatibility."""

    return term.whnf()


def reduce_inside_step(term: Term, red: Reducer) -> Term:
    """Wrapper around ``Term.reduce_inside_step`` for backwards compatibility."""

    return term.reduce_inside_step(red)


__all__ = ["whnf", "reduce_inside_step"]
