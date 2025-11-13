"""Type guard helpers leveraging modern typing features."""

from __future__ import annotations

from typing import TypeIs

from .ast import NatType, Pi, Term, Univ


def is_pi(term: Term) -> TypeIs[Pi]:
    """Return ``True`` if *term* is a :class:`~mltt.ast.Pi` type."""

    return isinstance(term, Pi)


def is_type_universe(term: Term) -> TypeIs[Univ]:
    """Return ``True`` if *term* is the :class:`~mltt.ast.Univ`."""

    return isinstance(term, Univ)


def is_nat_type(term: Term) -> TypeIs[NatType]:
    """Return ``True`` if *term* is :class:`~mltt.ast.NatType`."""

    return isinstance(term, NatType)


__all__ = ["is_pi", "is_type_universe", "is_nat_type"]
