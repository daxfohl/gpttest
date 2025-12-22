"""Sorted list predicate."""

from __future__ import annotations

from .list import ConsCtor, List, NilCtor
from ..core.ast import App, Pi, Term, Univ, Var
from ..core.debruijn import mk_app, Telescope, ArgList
from ..core.ind import Elim, Ctor, Ind

Sorted = Ind(
    name="Sorted",
    param_types=Telescope.of(
        Univ(0),  # A : Type
        Pi(Var(0), Pi(Var(1), Univ(0))),  # R : A -> A -> Type
    ),
    index_types=Telescope.of(App(List, Var(1))),  # xs : List A
    level=0,
)

SortedNilCtor = Ctor(
    name="sorted_nil",
    inductive=Sorted,
    result_indices=ArgList.of(App(NilCtor, Var(1))),
)

SortedOneCtor = Ctor(
    name="sorted_one",
    inductive=Sorted,
    field_schemas=Telescope.of(Var(1)),  # x : A
    result_indices=ArgList.of(
        mk_app(ConsCtor, Var(2), Var(0), App(NilCtor, Var(2)))
    ),  # [x]
)

SortedConsCtor = Ctor(
    name="sorted_cons",
    inductive=Sorted,
    field_schemas=Telescope.of(
        mk_app(List, Var(1)),  # xs : List A
        Var(2),  # x : A
        Var(3),  # y : A
        mk_app(Var(3), Var(1), Var(0)),  # R x y
        mk_app(  # ih : Sorted A R (y :: xs)
            Sorted,
            Var(5),
            Var(4),
            mk_app(ConsCtor, Var(5), Var(1), Var(3)),
        ),
    ),
    result_indices=ArgList.of(
        mk_app(  # x :: y :: xs
            ConsCtor,
            Var(6),
            Var(3),
            mk_app(ConsCtor, Var(6), Var(2), Var(4)),
        ),
    ),
)

object.__setattr__(
    Sorted, "constructors", (SortedNilCtor, SortedOneCtor, SortedConsCtor)
)


def SortedType(A: Term, R: Term, xs: Term) -> Term:
    return mk_app(Sorted, A, R, xs)


def SortedNil(A: Term, R: Term) -> Term:
    return mk_app(SortedNilCtor, A, R)


def SortedOne(A: Term, R: Term, x: Term) -> Term:
    return mk_app(SortedOneCtor, A, R, x)


def SortedCons(
    A: Term, R: Term, xs: Term, x: Term, y: Term, rel: Term, ih: Term
) -> Term:
    return mk_app(SortedConsCtor, A, R, xs, x, y, rel, ih)


def SortedRec(
    motive: Term, nil_case: Term, one_case: Term, cons_case: Term, proof: Term
) -> Elim:
    return Elim(
        inductive=Sorted,
        motive=motive,
        cases=(nil_case, one_case, cons_case),
        scrutinee=proof,
    )


__all__ = [
    "Sorted",
    "SortedType",
    "SortedNil",
    "SortedOne",
    "SortedCons",
    "SortedRec",
    "SortedNilCtor",
    "SortedOneCtor",
    "SortedConsCtor",
]
