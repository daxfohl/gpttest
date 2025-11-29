"""Helpers for dependent length-indexed vectors."""

from __future__ import annotations

from ..core.ast import (
    App,
    InductiveConstructor,
    InductiveElim,
    InductiveType,
    Term,
    Univ,
    Var,
)
from .nat import NatType, Succ, Zero

Vec = InductiveType(
    name="Vec", param_types=(Univ(0),), index_types=(NatType(),), level=0
)
NilCtor = InductiveConstructor("Nil", Vec, (), (Zero(),))
ConsCtor = InductiveConstructor(
    "Cons",
    Vec,
    (
        Var(1),  # head : A
        App(App(Vec, Var(2)), Var(1)),  # tail : Vec A n
    ),
    (Succ(Var(2)),),  # result index = Succ n
)
object.__setattr__(Vec, "constructors", (NilCtor, ConsCtor))


def VecType(elem_ty: Term, length: Term) -> App:
    return App(App(Vec, elem_ty), length)


def Nil(elem_ty: Term) -> App:
    return App(App(NilCtor, elem_ty), Zero())


def Cons(elem_ty: Term, n: Term, head: Term, tail: Term) -> Term:
    return App(App(App(App(ConsCtor, elem_ty), n), head), tail)


def VecRec(elem_ty: Term, P: Term, base: Term, step: Term, xs: Term) -> InductiveElim:
    """Recursor for vectors."""

    return InductiveElim(
        inductive=Vec,
        motive=P,
        cases=[
            base,
            step,
        ],
        scrutinee=xs,
    )
