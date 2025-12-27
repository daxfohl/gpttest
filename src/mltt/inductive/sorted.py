"""Sorted list predicate."""

from __future__ import annotations

from functools import cache

from .list import ConsCtorAt, ListAt, NilCtorAt
from ..core.ast import (
    App,
    ConstLevel,
    LevelExpr,
    LevelLike,
    LevelVar,
    Pi,
    Term,
    Univ,
    Var,
)
from ..core.debruijn import ArgList, Telescope, decompose_app, mk_app
from ..core.ind import Ctor, Elim, Ind


@cache
def _sorted_family(level: LevelExpr) -> tuple[Ind, Ctor, Ctor, Ctor]:
    list_ind = ListAt(level)
    nil_ctor = NilCtorAt(level)
    cons_ctor = ConsCtorAt(level)
    sorted_ind = Ind(
        name="Sorted",
        param_types=Telescope.of(
            Univ(level),  # A : Type
            Pi(Var(0), Pi(Var(1), Univ(level))),  # R : A -> A -> Type
        ),
        index_types=Telescope.of(App(list_ind, Var(1))),  # xs : List A
        level=level,
    )
    sorted_nil = Ctor(
        name="sorted_nil",
        inductive=sorted_ind,
        result_indices=ArgList.of(App(nil_ctor, Var(1))),
    )
    sorted_one = Ctor(
        name="sorted_one",
        inductive=sorted_ind,
        field_schemas=Telescope.of(Var(1)),  # x : A
        result_indices=ArgList.of(
            mk_app(cons_ctor, Var(2), Var(0), App(nil_ctor, Var(2)))
        ),  # [x]
    )
    sorted_cons = Ctor(
        name="sorted_cons",
        inductive=sorted_ind,
        field_schemas=Telescope.of(
            mk_app(list_ind, Var(1)),  # xs : List A
            Var(2),  # x : A
            Var(3),  # y : A
            mk_app(Var(3), Var(1), Var(0)),  # R x y
            mk_app(  # ih : Sorted A R (y :: xs)
                sorted_ind,
                Var(5),
                Var(4),
                mk_app(cons_ctor, Var(5), Var(1), Var(3)),
            ),
        ),
        result_indices=ArgList.of(
            mk_app(  # x :: y :: xs
                cons_ctor,
                Var(6),
                Var(3),
                mk_app(cons_ctor, Var(6), Var(2), Var(4)),
            ),
        ),
    )
    object.__setattr__(
        sorted_ind, "constructors", (sorted_nil, sorted_one, sorted_cons)
    )
    return sorted_ind, sorted_nil, sorted_one, sorted_cons


def _normalize_level(level: LevelLike) -> LevelExpr:
    if isinstance(level, LevelExpr):
        return level
    return ConstLevel(level)


Sorted, SortedNilCtor, SortedOneCtor, SortedConsCtor = _sorted_family(LevelVar(0))


def SortedAt(level: LevelLike) -> Ind:
    return _sorted_family(_normalize_level(level))[0]


def SortedNilCtorAt(level: LevelLike) -> Ctor:
    return _sorted_family(_normalize_level(level))[1]


def SortedOneCtorAt(level: LevelLike) -> Ctor:
    return _sorted_family(_normalize_level(level))[2]


def SortedConsCtorAt(level: LevelLike) -> Ctor:
    return _sorted_family(_normalize_level(level))[3]


def SortedType(A: Term, R: Term, xs: Term, *, level: LevelLike | None = None) -> Term:
    if level is None:
        level = A.expect_universe()
    return mk_app(SortedAt(level), A, R, xs)


def SortedNil(A: Term, R: Term, *, level: LevelLike | None = None) -> Term:
    if level is None:
        level = A.expect_universe()
    return mk_app(SortedNilCtorAt(level), A, R)


def SortedOne(A: Term, R: Term, x: Term, *, level: LevelLike | None = None) -> Term:
    if level is None:
        level = A.expect_universe()
    return mk_app(SortedOneCtorAt(level), A, R, x)


def SortedCons(
    A: Term,
    R: Term,
    xs: Term,
    x: Term,
    y: Term,
    rel: Term,
    ih: Term,
    *,
    level: LevelLike | None = None,
) -> Term:
    if level is None:
        level = A.expect_universe()
    return mk_app(SortedConsCtorAt(level), A, R, xs, x, y, rel, ih)


def SortedRec(
    motive: Term,
    nil_case: Term,
    one_case: Term,
    cons_case: Term,
    proof: Term,
    *,
    level: LevelLike | None = None,
) -> Elim:
    if level is None:
        scrut_ty = proof.infer_type().whnf()
        head, _ = decompose_app(scrut_ty)
        if not isinstance(head, Ind):
            raise TypeError(f"SortedRec scrutinee is not a Sorted: {scrut_ty}")
        inductive = head
    else:
        inductive = SortedAt(level)
    return Elim(
        inductive=inductive,
        motive=motive,
        cases=(nil_case, one_case, cons_case),
        scrutinee=proof,
    )


__all__ = [
    "Sorted",
    "SortedAt",
    "SortedType",
    "SortedNil",
    "SortedOne",
    "SortedCons",
    "SortedRec",
    "SortedNilCtor",
    "SortedNilCtorAt",
    "SortedOneCtor",
    "SortedOneCtorAt",
    "SortedConsCtor",
    "SortedConsCtorAt",
]
