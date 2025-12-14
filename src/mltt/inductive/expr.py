"""Typed expression GADT indexed by its result type."""

from __future__ import annotations

from .sigma import Sigma
from ..core.ast import Ctor, Elim, I, Lam, Term, Univ, Var
from ..core.debruijn import shift
from ..core.inductive_utils import apply_term

# Expr (Ty : Type) (τ : Ty) : Type
Expr = I(name="Expr", param_types=(Univ(0),), index_types=(Var(0),), level=0)

ConstCtor = Ctor(
    name="const",
    inductive=Expr,
    arg_types=(Var(0),),  # value : τ
    result_indices=(Var(1),),  # τ
)

PairCtor = Ctor(
    name="pair",
    inductive=Expr,
    arg_types=(
        Var(1),  # A : Ty
        Var(2),  # B : Ty
        apply_term(Expr, Var(3), Var(1)),  # Expr Ty A
        apply_term(Expr, Var(4), Var(1)),  # Expr Ty B
    ),
    result_indices=(
        apply_term(  # A × B as a Sigma with constant second component.
            Sigma,
            Var(3),  # A
            Lam(Var(3), shift(Var(2), 1)),  # λ_:A. B
        ),
    ),
)

object.__setattr__(Expr, "constructors", (ConstCtor, PairCtor))


def ExprType(Ty: Term, tau: Term) -> Term:
    return apply_term(Expr, Ty, tau)


def Const(Ty: Term, tau: Term, value: Term) -> Term:
    return apply_term(ConstCtor, Ty, tau, value)


def Pair(Ty: Term, A: Term, B: Term, lhs: Term, rhs: Term) -> Term:
    pair_index = apply_term(Sigma, A, Lam(A, shift(B, 1)))
    return apply_term(PairCtor, Ty, pair_index, A, B, lhs, rhs)


def ExprRec(motive: Term, const_case: Term, pair_case: Term, scrutinee: Term) -> Elim:
    return Elim(
        inductive=Expr,
        motive=motive,
        cases=(const_case, pair_case),
        scrutinee=scrutinee,
    )


__all__ = [
    "Expr",
    "ExprType",
    "Const",
    "Pair",
    "ExprRec",
    "ConstCtor",
    "PairCtor",
]
