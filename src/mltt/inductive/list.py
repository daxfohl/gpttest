"""Helpers for building generic list terms and combinators."""

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

List = I(name="List", param_types=(Univ(0),), level=0)
NilCtor = Ctor("Nil", List, ())
ConsCtor = Ctor(
    "Cons",
    List,
    (
        Var(0),
        App(Var(1), List),
    ),
)
object.__setattr__(List, "constructors", (NilCtor, ConsCtor))


def ListType(elem_ty: Term) -> App:
    return App(elem_ty, List)


def Nil(elem_ty: Term) -> App:
    return App(elem_ty, NilCtor)


def Cons(elem_ty: Term, head: Term, tail: Term) -> Term:
    return App(tail, App(head, App(elem_ty, ConsCtor)))


def ListRec(P: Term, base: Term, step: Term, xs: Term) -> Elim:
    """Recursor for ``List elem_ty`` using the generic eliminator."""

    return Elim(
        inductive=List,
        motive=P,
        cases=(base, step),
        scrutinee=xs,
    )
