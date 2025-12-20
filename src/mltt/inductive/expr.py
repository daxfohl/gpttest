"""Typed expression GADT indexed by its result type."""

from __future__ import annotations

from .sigma import Sigma
from ..core.ast import Lam, Term, Univ, Var
from ..core.debruijn import mk_app
from ..core.ind import Elim, Ctor, Ind

# Expr (Ty : Type) (τ : Ty) : Type
Expr = Ind(name="Expr", param_types=(Univ(0),), index_types=(Var(0),), level=0)

ConstCtor = Ctor(
    name="const",
    inductive=Expr,
    arg_types=(
        Var(0),  # τ : Ty
        Var(0),  # value : τ
    ),
    result_indices=(Var(1),),  # τ
)

PairCtor = Ctor(
    name="pair",
    inductive=Expr,
    arg_types=(
        Var(0),  # A : Ty
        Var(1),  # B : Ty
        mk_app(Expr, Var(2), Var(1)),  # Expr Ty A
        mk_app(Expr, Var(3), Var(1)),  # Expr Ty B
    ),
    result_indices=(
        mk_app(  # A × B as a Sigma with constant second component.
            Sigma,
            Var(3),  # A
            Lam(Var(3), Var(2).shift(1)),  # λ_:A. B
        ),
    ),
)

object.__setattr__(Expr, "constructors", (ConstCtor, PairCtor))


def ExprType(Ty: Term, tau: Term) -> Term:
    return mk_app(Expr, Ty, tau)


def Const(Ty: Term, tau: Term, value: Term) -> Term:
    return mk_app(ConstCtor, Ty, tau, value)


def Pair(Ty: Term, A: Term, B: Term, lhs: Term, rhs: Term) -> Term:
    return mk_app(PairCtor, Ty, A, B, lhs, rhs)


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
