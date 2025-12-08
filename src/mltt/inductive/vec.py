"""Helpers for dependent length-indexed vectors."""

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

Vec = I(name="Vec", param_types=(Univ(0),), index_types=(NatType(),), level=0)
NilCtor = Ctor("Nil", Vec, (), (Zero(),))
ConsCtor = Ctor(
    "Cons",
    Vec,
    (
        Var(1),  # head : A
        App(Var(1), App(Var(2), Vec)),  # tail : Vec A n
    ),
    (Succ(Var(2)),),  # result index = Succ n
)
object.__setattr__(Vec, "constructors", (NilCtor, ConsCtor))


def VecType(elem_ty: Term, length: Term) -> App:
    return App(length, App(elem_ty, Vec))


def Nil(elem_ty: Term) -> App:
    return App(Zero(), App(elem_ty, NilCtor))


def Cons(elem_ty: Term, n: Term, head: Term, tail: Term) -> Term:
    return App(tail, App(head, App(n, App(elem_ty, ConsCtor))))


def VecRec(P: Term, base: Term, step: Term, xs: Term) -> Elim:
    """Recursor for vectors."""

    return Elim(
        inductive=Vec,
        motive=P,
        cases=(base, step),
        scrutinee=xs,
    )
