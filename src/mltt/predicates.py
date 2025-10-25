"""Type guard helpers leveraging modern typing features.

The helpers here are tiny runtime wrappers around ``isinstance`` checks, but
annotated with :data:`typing.TypeIs` so static analysis knows that a successful
check refines the operand's type.  Python 3.14 introduces ``TypeIs`` (PEP 742)
and the compatibility layer in :mod:`mltt._compat` allows us to use it today
without giving up support for older interpreters.
"""

from __future__ import annotations

from ._compat import TypeIs
from .ast import NatType, Pi, Sigma, Term, TypeUniverse


def is_pi(term: Term) -> TypeIs[Pi]:
    """Return ``True`` if *term* is a :class:`~mltt.ast.Pi` type."""

    return isinstance(term, Pi)


def is_sigma(term: Term) -> TypeIs[Sigma]:
    """Return ``True`` if *term* is a :class:`~mltt.ast.Sigma` type."""

    return isinstance(term, Sigma)


def is_type_universe(term: Term) -> TypeIs[TypeUniverse]:
    """Return ``True`` if *term* is the :class:`~mltt.ast.TypeUniverse`."""

    return isinstance(term, TypeUniverse)


def is_nat_type(term: Term) -> TypeIs[NatType]:
    """Return ``True`` if *term* is :class:`~mltt.ast.NatType`."""

    return isinstance(term, NatType)


__all__ = ["is_pi", "is_sigma", "is_type_universe", "is_nat_type"]

