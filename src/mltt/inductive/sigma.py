"""Helpers for dependent pairs (Sigma type)."""

from __future__ import annotations

from ..core.ast import (
    App,
    Ctor,
    Elim,
    I,
    Pi,
    Term,
    Univ,
    Var,
)

Sigma = I(
    name="Sigma",
    param_types=(
        Univ(0),  # A : Type
        Pi(Var(0), Univ(0)),  # B : A -> Type
    ),
    level=0,
)
PairCtor = Ctor(
    "Pair",
    Sigma,
    (
        Var(1),  # a : A
        App(Var(0), Var(1)),  # b : B a
    ),
)
object.__setattr__(Sigma, "constructors", (PairCtor,))


def SigmaType(A: Term, B: Term) -> App:
    return App(B, App(A, Sigma))


def Pair(A: Term, B: Term, a: Term, b: Term) -> Term:
    return App(b, App(a, App(B, App(A, PairCtor))))


def SigmaRec(P: Term, pair_case: Term, pair: Term) -> Elim:
    """Recursor for ``Sigma A B`` using the generic eliminator."""

    return Elim(
        inductive=Sigma,
        motive=P,
        cases=(pair_case,),
        scrutinee=pair,
    )
