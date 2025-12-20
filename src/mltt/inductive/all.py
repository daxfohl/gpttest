"""All elements of a list satisfy a predicate."""

from __future__ import annotations

from .list import ConsCtor, List, NilCtor
from ..core.ast import App, Pi, Term, Univ, Var
from ..core.debruijn import mk_app
from ..core.ind import Elim, Ctor, Ind

All = Ind(
    name="All",
    param_types=(
        Univ(0),  # A : Type
        Pi(Var(0), Univ(0)),  # P : A -> Type
    ),
    index_types=(App(List, Var(1)),),  # xs : List A
    level=0,
)

AllNilCtor = Ctor(
    name="all_nil",
    inductive=All,
    arg_types=(),
    result_indices=(App(NilCtor, Var(1)),),
)

AllConsCtor = Ctor(
    name="all_cons",
    inductive=All,
    arg_types=(
        mk_app(List, Var(1)),  # xs : List A
        Var(2),  # x : A
        mk_app(Var(2), Var(0)),  # px : P x
        mk_app(All, Var(4), Var(3), Var(2)),  # ih : All A P xs
    ),
    result_indices=(mk_app(ConsCtor, Var(5), Var(2), Var(3)),),  # x :: xs
)

object.__setattr__(All, "constructors", (AllNilCtor, AllConsCtor))


def AllType(A: Term, P: Term, xs: Term) -> Term:
    return mk_app(All, A, P, xs)


def AllNil(A: Term, P: Term) -> Term:
    return mk_app(AllNilCtor, A, P)


def AllCons(A: Term, P: Term, xs: Term, x: Term, px: Term, ih: Term) -> Term:
    return mk_app(AllConsCtor, A, P, xs, x, px, ih)


def AllRec(motive: Term, nil_case: Term, cons_case: Term, proof: Term) -> Elim:
    return Elim(
        inductive=All, motive=motive, cases=(nil_case, cons_case), scrutinee=proof
    )


__all__ = ["All", "AllType", "AllNil", "AllCons", "AllRec"]
