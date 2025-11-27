"""Finite ordinals ``Fin n`` (natural numbers strictly less than ``n``)."""

from __future__ import annotations

from .ast import (
    App,
    InductiveConstructor,
    InductiveElim,
    InductiveType,
    Term,
    Univ,
    Var,
)
from .nat import NatType, Succ, Zero

Fin = InductiveType(index_types=(NatType(),), level=0)
FZCtor = InductiveConstructor(Fin, (), (Succ(Var(0)),))
FSCtor = InductiveConstructor(
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


def FinRec(P: Term, base: Term, step: Term, k: Term) -> InductiveElim:
    """Recursor for ``Fin`` expressed via the generalized eliminator."""

    return InductiveElim(
        inductive=Fin,
        motive=P,
        cases=[base, step],
        scrutinee=k,
    )
