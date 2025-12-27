"""All elements of a list satisfy a predicate."""

from __future__ import annotations

from functools import cache

from .list import ConsCtorAt, ListAt, NilCtorAt
from ..core.ast import App, ConstLevel, LevelExpr, LevelLike, Pi, Term, Univ, Var
from ..core.debruijn import ArgList, Telescope, mk_app
from ..core.ind import Ctor, Elim, Ind


@cache
def _all_family_norm(level: LevelExpr) -> tuple[Ind, Ctor, Ctor]:
    list_ind = ListAt(level)
    nil_ctor = NilCtorAt(level)
    cons_ctor = ConsCtorAt(level)
    all_ind = Ind(
        name="All",
        param_types=Telescope.of(
            Univ(level),  # A : Type
            Pi(Var(0), Univ(level)),  # P : A -> Type
        ),
        index_types=Telescope.of(App(list_ind, Var(1))),  # xs : List A
        level=level,
    )
    all_nil = Ctor(
        name="all_nil",
        inductive=all_ind,
        result_indices=ArgList.of(App(nil_ctor, Var(1))),
    )
    all_cons = Ctor(
        name="all_cons",
        inductive=all_ind,
        field_schemas=Telescope.of(
            mk_app(list_ind, Var(1)),  # xs : List A
            Var(2),  # x : A
            mk_app(Var(2), Var(0)),  # px : P x
            mk_app(all_ind, Var(4), Var(3), Var(2)),  # ih : All A P xs
        ),
        result_indices=ArgList.of(mk_app(cons_ctor, Var(5), Var(2), Var(3))),  # x :: xs
    )
    object.__setattr__(all_ind, "constructors", (all_nil, all_cons))
    return all_ind, all_nil, all_cons


def _normalize_level(level: LevelLike) -> LevelExpr:
    if isinstance(level, LevelExpr):
        return level
    return ConstLevel(level)


def _all_family(level: LevelLike) -> tuple[Ind, Ctor, Ctor]:
    return _all_family_norm(_normalize_level(level))


All, AllNilCtor, AllConsCtor = _all_family(0)


def AllAt(level: LevelLike) -> Ind:
    return _all_family(level)[0]


def AllNilCtorAt(level: LevelLike) -> Ctor:
    return _all_family(level)[1]


def AllConsCtorAt(level: LevelLike) -> Ctor:
    return _all_family(level)[2]


def AllType(A: Term, P: Term, xs: Term, *, level: LevelLike = 0) -> Term:
    return mk_app(AllAt(level), A, P, xs)


def AllNil(A: Term, P: Term, *, level: LevelLike = 0) -> Term:
    return mk_app(AllNilCtorAt(level), A, P)


def AllCons(
    A: Term,
    P: Term,
    xs: Term,
    x: Term,
    px: Term,
    ih: Term,
    *,
    level: LevelLike = 0,
) -> Term:
    return mk_app(AllConsCtorAt(level), A, P, xs, x, px, ih)


def AllRec(
    motive: Term,
    nil_case: Term,
    cons_case: Term,
    proof: Term,
    *,
    level: LevelLike = 0,
) -> Elim:
    inductive = AllAt(level)
    return Elim(
        inductive=inductive, motive=motive, cases=(nil_case, cons_case), scrutinee=proof
    )


__all__ = [
    "All",
    "AllAt",
    "AllType",
    "AllNil",
    "AllCons",
    "AllRec",
    "AllNilCtor",
    "AllNilCtorAt",
    "AllConsCtor",
    "AllConsCtorAt",
]
