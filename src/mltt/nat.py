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


add = Lam(
    NatType(),
    Lam(
        NatType(),
        NatRec(
            P=Lam(NatType(), NatType()),
            z=Var(1),
            s=Lam(
                NatType(),
                Lam(NatType(), Succ(Var(0))),
            ),
            n=Var(0),
        ),
    ),
)


def add_zero_right() -> Term:
    """Proof that ``add n 0`` is definitionally equal to ``n`` for all ``n``."""

    return Lam(
        NatType(),
        Refl(NatType(), App(App(add, Var(0)), Zero())),
    )


def add_succ_right() -> Term:
    """Proof that ``add m (Succ n)`` equals ``Succ (add m n)`` for all naturals."""

    return Lam(
        NatType(),
        Lam(
            NatType(),
            Refl(NatType(), App(App(add, Var(1)), Succ(Var(0)))),
        ),
    )
