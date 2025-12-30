"""Typed expression GADT indexed by its result type."""

from __future__ import annotations

from mltt.inductive.sigma import Sigma
from mltt.kernel.ast import Lam, Term, Univ, Var
from mltt.kernel.telescope import mk_app, Telescope, ArgList
from mltt.kernel.ind import Elim, Ctor, Ind

# Expr (Ty : Type1) (τ : Ty) : Type1
Expr = Ind(
    name="Expr",
    param_types=Telescope.of(Univ(1)),
    index_types=Telescope.of(Var(0)),
    level=1,
)

ConstCtor = Ctor(
    name="const",
    inductive=Expr,
    field_schemas=Telescope.of(
        Var(0),  # τ : Ty
        Var(0),  # value : τ
    ),
    result_indices=ArgList.of(Var(1)),  # τ
)

PairCtor = Ctor(
    name="pair",
    inductive=Expr,
    field_schemas=Telescope.of(
        Var(0),  # A : Ty
        Var(1),  # B : Ty
        mk_app(Expr, Var(2), Var(1)),  # Expr Ty A
        mk_app(Expr, Var(3), Var(1)),  # Expr Ty B
    ),
    result_indices=ArgList.of(
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


def ExprElim(motive: Term, const_case: Term, pair_case: Term, scrutinee: Term) -> Elim:
    return Elim(
        inductive=Expr,
        motive=motive,
        cases=(const_case, pair_case),
        scrutinee=scrutinee,
    )
