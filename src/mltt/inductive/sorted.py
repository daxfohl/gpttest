"""Sorted list predicate."""

from __future__ import annotations

from mltt.inductive.list import ConsCtorAt, ListAt, NilCtorAt
from mltt.kernel.ast import App, Pi, Term, Univ, Var, UApp
from mltt.kernel.ind import Elim, Ctor, Ind
from mltt.kernel.levels import LVar, LevelExpr
from mltt.kernel.telescope import mk_app, Telescope, ArgList


def _sorted() -> tuple[Ind, Ctor, Ctor, Ctor]:
    u = LVar(0)
    list_ind = ListAt(u)
    nil_ctor = NilCtorAt(u)
    cons_ctor = ConsCtorAt(u)
    sorted_ind = Ind(
        name="Sorted",
        uarity=1,
        param_types=Telescope.of(
            Univ(u),  # A : Type
            Pi(Var(0), Pi(Var(1), Univ(u))),  # R : A -> A -> Type
        ),
        index_types=Telescope.of(App(list_ind, Var(1))),  # xs : List A
        level=u,
    )

    sorted_nil_ctor = Ctor(
        name="sorted_nil",
        inductive=sorted_ind,
        result_indices=ArgList.of(App(nil_ctor, Var(1))),
        uarity=1,
    )

    sorted_one_ctor = Ctor(
        name="sorted_one",
        inductive=sorted_ind,
        field_schemas=Telescope.of(Var(1)),  # x : A
        result_indices=ArgList.of(
            mk_app(cons_ctor, Var(2), Var(0), App(nil_ctor, Var(2)))
        ),  # [x]
        uarity=1,
    )

    sorted_cons_ctor = Ctor(
        name="sorted_cons",
        inductive=sorted_ind,
        field_schemas=Telescope.of(
            mk_app(list_ind, Var(1)),  # xs : List A
            Var(2),  # x : A
            Var(3),  # y : A
            mk_app(Var(3), Var(1), Var(0)),  # R x y
            mk_app(  # ih : Sorted A R (y :: xs)
                UApp(sorted_ind, u),
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
        uarity=1,
    )

    object.__setattr__(
        sorted_ind, "constructors", (sorted_nil_ctor, sorted_one_ctor, sorted_cons_ctor)
    )
    return sorted_ind, sorted_nil_ctor, sorted_one_ctor, sorted_cons_ctor


Sorted_U, SortedNil_U, SortedOne_U, SortedCons_U = _sorted()


def SortedAt(level: LevelExpr | int = 0) -> Term:
    return UApp(Sorted_U, level)


def SortedNilCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(SortedNil_U, level)


def SortedOneCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(SortedOne_U, level)


def SortedConsCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(SortedCons_U, level)


Sorted = SortedAt()
SortedNilCtor = SortedNilCtorAt()
SortedOneCtor = SortedOneCtorAt()
SortedConsCtor = SortedConsCtorAt()


def SortedType(A: Term, R: Term, xs: Term, *, level: LevelExpr | int = 0) -> Term:
    return mk_app(SortedAt(level), A, R, xs)


def SortedNil(A: Term, R: Term, *, level: LevelExpr | int = 0) -> Term:
    return mk_app(SortedNilCtorAt(level), A, R)


def SortedOne(A: Term, R: Term, x: Term, *, level: LevelExpr | int = 0) -> Term:
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
    level: LevelExpr | int = 0,
) -> Term:
    return mk_app(SortedConsCtorAt(level), A, R, xs, x, y, rel, ih)


def SortedRec(
    motive: Term, nil_case: Term, one_case: Term, cons_case: Term, proof: Term
) -> Elim:
    return Elim(
        inductive=Sorted_U,
        motive=motive,
        cases=(nil_case, one_case, cons_case),
        scrutinee=proof,
    )
