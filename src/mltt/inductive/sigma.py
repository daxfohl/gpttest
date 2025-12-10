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
from ..core.inductive_utils import apply_term

Sigma = I(
    name="Sigma",
    param_types=(
        Univ(0),  # A : Type
        Pi(Var(0), Univ(0)),  # B : A -> Type
    ),
    level=0,
)
PairCtor = Ctor(
    name="Pair",
    inductive=Sigma,
    arg_types=(
        Var(1),  # a : A
        App(Var(1), Var(0)),  # b : B a
    ),
)
object.__setattr__(Sigma, "constructors", (PairCtor,))


def SigmaType(A: Term, B: Term) -> Term:
    return apply_term(Sigma, A, B)


def Pair(A: Term, B: Term, a: Term, b: Term) -> Term:
    return apply_term(PairCtor, A, B, a, b)


def SigmaRec(P: Term, pair_case: Term, pair: Term) -> Elim:
    """Recursor for ``Sigma A B`` using the generic eliminator."""

    return Elim(
        inductive=Sigma,
        motive=P,
        cases=(pair_case,),
        scrutinee=pair,
    )
