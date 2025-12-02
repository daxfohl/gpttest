"""Finite ordinals ``Fin n`` (natural numbers strictly less than ``n``)."""

from __future__ import annotations

from ..core.ast import (
    App,
    Ctor,
    Elim,
    I,
    Term,
    Univ,
    Var,
)
from .nat import NatType, Succ, Zero

Fin = I(name="Fin", index_types=(NatType(),), level=0)
FZCtor = Ctor("FZ", Fin, (), (Succ(Var(0)),))
FSCtor = Ctor(
    "FS",
    Fin,
    (App(Fin, Var(0)),),
    (Succ(Var(1)),),
)
object.__setattr__(Fin, "constructors", (FZCtor, FSCtor))


def FinType(n: Term) -> App:
    return App(Fin, n)


def FZ(n: Term) -> App:
    return App(FZCtor, n)


def FS(n: Term, k: Term) -> Term:
    return App(App(FSCtor, n), k)


def FinRec(P: Term, base: Term, step: Term, k: Term) -> Elim:
    """Recursor for ``Fin`` expressed via the generalized eliminator."""

    return Elim(
        inductive=Fin,
        motive=P,
        cases=[base, step],
        scrutinee=k,
    )
