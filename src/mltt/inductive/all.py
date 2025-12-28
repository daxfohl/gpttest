"""All elements of a list satisfy a predicate."""

from __future__ import annotations

from functools import cache

from mltt.inductive.list import ConsCtorAt, ListAt, NilCtorAt
from mltt.core.ast import App, Pi, Term, Univ, Var
from mltt.core.debruijn import mk_app, Telescope, ArgList
from mltt.core.ind import Elim, Ctor, Ind


@cache
def _all_family(level: int) -> tuple[Ind, Ctor, Ctor]:
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

    all_nil_ctor = Ctor(
        name="all_nil",
        inductive=all_ind,
        result_indices=ArgList.of(App(nil_ctor, Var(1))),
    )

    all_cons_ctor = Ctor(
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

    object.__setattr__(all_ind, "constructors", (all_nil_ctor, all_cons_ctor))
    return all_ind, all_nil_ctor, all_cons_ctor


All, AllNilCtor, AllConsCtor = _all_family(0)


def AllAt(level: int = 0) -> Ind:
    return _all_family(level)[0]


def AllNilCtorAt(level: int = 0) -> Ctor:
    return _all_family(level)[1]


def AllConsCtorAt(level: int = 0) -> Ctor:
    return _all_family(level)[2]


def AllType(A: Term, P: Term, xs: Term, *, level: int = 0) -> Term:
    return mk_app(AllAt(level), A, P, xs)


def AllNil(A: Term, P: Term, *, level: int = 0) -> Term:
    return mk_app(AllNilCtorAt(level), A, P)


def AllCons(
    A: Term, P: Term, xs: Term, x: Term, px: Term, ih: Term, *, level: int = 0
) -> Term:
    return mk_app(AllConsCtorAt(level), A, P, xs, x, px, ih)


def AllRec(
    motive: Term, nil_case: Term, cons_case: Term, proof: Term, *, level: int = 0
) -> Elim:
    return Elim(
        inductive=AllAt(level),
        motive=motive,
        cases=(nil_case, cons_case),
        scrutinee=proof,
    )
