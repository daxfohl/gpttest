"""Helpers for building natural number terms and common combinators."""

from __future__ import annotations

from .ast import App, Id, Lam, NatRec, NatType, Pi, Refl, Succ, Term, Var, Zero
from .eq import cong, sym, trans
from .eval import normalize
from .typing import type_check


def numeral(value: int) -> Term:
    """Return the canonical term representing the natural number ``value``."""

    term: Term = Zero()
    for _ in range(value):
        term = Succ(term)
    return term


def add():
    """
add : Nat → Nat → Nat
Addition by recursion on the first argument.

add a b =
  NatRec (λ_. Nat) b (λ_ r. Succ r) a

Rules:
  add Zero b = b
  add (Succ a) b = Succ (add a b)
"""
    return Lam(
        NatType(),
        Lam(
            NatType(),
            NatRec(
                P=Lam(NatType(), NatType()),
                z=Var(0),
                s=Lam(
                    NatType(),
                    Lam(NatType(), Succ(Var(0))),
                ),
                n=Var(1),
            ),
        ),
    )
