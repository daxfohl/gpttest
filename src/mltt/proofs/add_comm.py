"""Proofs around addition on natural numbers."""

from __future__ import annotations

from ..core.ast import Lam, Term, Var
from ..core.inductive_utils import nested_lam
from ..inductive.eq import Refl
from ..inductive.nat import NatType, Succ, Zero, add_terms, add_n_0


def add_zero_right() -> Term:
    """∀ n. add n 0 = n."""

    return add_n_0()


def add_zero_left() -> Term:
    """∀ n. add 0 n = n."""

    return Lam(
        NatType(),
        Refl(NatType(), add_terms(Zero(), Var(0))),
    )


def succ_add() -> Term:
    """∀ n m. add (Succ n) m = Succ (add n m)."""

    return nested_lam(
        NatType(),
        NatType(),
        body=Refl(NatType(), add_terms(Succ(Var(1)), Var(0))),
    )


__all__ = [
    "add_zero_right",
    "add_zero_left",
    "succ_add",
]
