"""Helpers for dependent length-indexed vectors."""

from __future__ import annotations

from .nat import NatType, Succ, Zero
from ..core.ast import (
    Ctor,
    Elim,
    I,
    Term,
    Univ,
    Var,
)
from ..core.inductive_utils import apply_term

Vec = I(name="Vec", param_types=(Univ(0),), index_types=(NatType(),), level=0)
NilCtor = Ctor(
    name="Nil",
    inductive=Vec,
    arg_types=(),
    result_indices=(Zero(),),
)
ConsCtor = Ctor(
    name="Cons",
    inductive=Vec,
    arg_types=(
        NatType(),  # n : Nat
        Var(1),  # head : A
        apply_term(Vec, Var(2), Var(1)),  # tail : Vec A n
    ),
    result_indices=(Succ(Var(2)),),  # result index = Succ n
)
object.__setattr__(Vec, "constructors", (NilCtor, ConsCtor))


def VecType(elem_ty: Term, length: Term) -> Term:
    return apply_term(Vec, elem_ty, length)


def Nil(elem_ty: Term) -> Term:
    return apply_term(NilCtor, elem_ty)


def Cons(elem_ty: Term, n: Term, head: Term, tail: Term) -> Term:
    return apply_term(ConsCtor, elem_ty, n, head, tail)


def VecRec(P: Term, base: Term, step: Term, xs: Term) -> Elim:
    """Recursor for vectors."""

    return Elim(
        inductive=Vec,
        motive=P,
        cases=(base, step),
        scrutinee=xs,
    )
