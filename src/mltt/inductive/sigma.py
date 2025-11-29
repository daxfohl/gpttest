"""Helpers for dependent pairs (Sigma type)."""

from __future__ import annotations

from ..core.ast import (
    App,
    InductiveConstructor,
    InductiveElim,
    InductiveType,
    Pi,
    Term,
    Univ,
    Var,
)

Sigma = InductiveType(
    name="Sigma",
    param_types=(
        Univ(0),  # A : Type
        Pi(Var(0), Univ(0)),  # B : A -> Type
    ),
    level=0,
)
PairCtor = InductiveConstructor(
    "Pair",
    Sigma,
    (
        Var(1),  # a : A
        App(Var(1), Var(0)),  # b : B a
    ),
)
object.__setattr__(Sigma, "constructors", (PairCtor,))


def SigmaType(A: Term, B: Term) -> App:
    return App(App(Sigma, A), B)


def Pair(A: Term, B: Term, a: Term, b: Term) -> Term:
    return App(App(App(App(PairCtor, A), B), a), b)


def SigmaRec(P: Term, pair_case: Term, pair: Term) -> InductiveElim:
    """Recursor for ``Sigma A B`` using the generic eliminator."""

    return InductiveElim(
        inductive=Sigma,
        motive=P,
        cases=[pair_case],
        scrutinee=pair,
    )
