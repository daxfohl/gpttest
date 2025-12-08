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
from ..core.inductive_utils import apply_term

List = I(name="List", param_types=(Univ(0),), level=0)
NilCtor = Ctor("Nil", List, ())
ConsCtor = Ctor(
    "Cons",
    List,
    (
        Var(0),
        App(List, Var(1)),
    ),
)
object.__setattr__(List, "constructors", (NilCtor, ConsCtor))


def ListType(elem_ty: Term) -> App:
    return App(List, elem_ty)


def Nil(elem_ty: Term) -> App:
    return App(NilCtor, elem_ty)


def Cons(elem_ty: Term, head: Term, tail: Term) -> Term:
    return apply_term(ConsCtor, elem_ty, head, tail)


def ListRec(P: Term, base: Term, step: Term, xs: Term) -> Elim:
    """Recursor for ``List elem_ty`` using the generic eliminator."""

    return Elim(
        inductive=List,
        motive=P,
        cases=(base, step),
        scrutinee=xs,
    )
