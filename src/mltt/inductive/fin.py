"""Finite ordinals ``Fin n`` (natural numbers strictly less than ``n``)."""

from __future__ import annotations

from ..core.ast import (
    App,
    Ctor,
    Elim,
    I,
    Term,
    Var,
)
from ..core.inductive_utils import apply_term
from .nat import NatType, Succ, numeral

Fin = I(name="Fin", index_types=(NatType(),), level=0)
FZCtor = Ctor(
    name="FZ",
    inductive=Fin,
    arg_types=(),
    result_indices=(Succ(Var(0)),),
)
FSCtor = Ctor(
    name="FS",
    inductive=Fin,
    arg_types=(App(Fin, Var(0)),),
    result_indices=(Succ(Var(1)),),
)
object.__setattr__(Fin, "constructors", (FZCtor, FSCtor))


def FinType(n: Term) -> Term:
    return App(Fin, n)


def FZ(n: Term) -> Term:
    return App(FZCtor, n)


def FS(n: Term, k: Term) -> Term:
    return apply_term(FSCtor, n, k)


def FinRec(P: Term, base: Term, step: Term, k: Term) -> Elim:
    """Recursor for ``Fin`` expressed via the generalized eliminator."""

    return Elim(
        inductive=Fin,
        motive=P,
        cases=(base, step),
        scrutinee=k,
    )


def of_int(i: int, n: int) -> Term:
    i %= n
    # n=1, i=0; n=2, i=[0, 1]
    t: Term = FZ(numeral(n - i - 1))
    # n=1 FZ 0 -> Fin 1; n=2 FZ [1, 0] -> Fin [2, 1]
    for j in range(i):
        t = FS(numeral(n - i + j), t)
        # n=2, i=1: j=0, FS 1 -> Fin 2
    return t
