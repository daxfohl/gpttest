"""Helpers for dependent length-indexed vectors."""

from __future__ import annotations

from .fin import FinType, FZ, FS
from .nat import NatType, Succ, Zero
from ..core.ast import Term, Univ, Var
from ..core.ind import Elim, Ctor, Ind
from ..core.util import apply_term, nested_lam

Vec = Ind(name="Vec", param_types=(Univ(0),), index_types=(NatType(),), level=0)
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


def vec_to_fin_term() -> Term:
    """
    Π A. Π n. Vec A n -> Fin (Succ n)

    Converts a length-indexed vector into an inhabitant of ``Fin (Succ n)`` by
    recursion on the vector, incrementing the induction hypothesis in the
    ``Cons`` branch.
    """

    motive = nested_lam(
        NatType(),  # n
        VecType(Var(3), Var(0)),  # xs : Vec A n (A is Var(3) in Γ,n,xs)
        body=FinType(Succ(Var(1))),  # Fin (Succ n)
    )
    step = nested_lam(
        NatType(),  # n
        Var(3),  # x : A
        VecType(Var(4), Var(1)),  # xs : Vec A n
        apply_term(motive.shift(2), Var(2), Var(0)),  # ih : P n xs
        body=FS(Succ(Var(3)), Var(0)),  # Fin (Succ (Succ n))
    )

    return nested_lam(
        Univ(0),  # A
        NatType(),  # n
        VecType(Var(1), Var(0)),  # xs : Vec A n
        body=VecRec(motive, FZ(Zero()), step, Var(0)),
    )


def to_fin(elem_ty: Term, length: Term, xs: Term) -> Term:
    """Apply ``vec_to_fin_term`` to concrete arguments."""

    return apply_term(vec_to_fin_term(), elem_ty, length, xs)
