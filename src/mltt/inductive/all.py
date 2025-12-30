"""All elements of a list satisfy a predicate."""

from __future__ import annotations

from mltt.inductive.list import ConsCtorAt, ListAt, NilCtorAt
from mltt.kernel.ast import App, Pi, Term, Univ, Var, UApp
from mltt.kernel.ind import Elim, Ctor, Ind
from mltt.kernel.levels import LVar, LevelExpr
from mltt.kernel.telescope import mk_app, mk_uapp, Telescope, ArgList


def _all() -> tuple[Ind, Ctor, Ctor]:
    u = LVar(0)
    list_ind = ListAt(u)
    nil_ctor = NilCtorAt(u)
    cons_ctor = ConsCtorAt(u)
    all_ind = Ind(
        name="All",
        uarity=1,
        param_types=Telescope.of(
            Univ(u),  # A : Type
            Pi(Var(0), Univ(u)),  # P : A -> Type
        ),
        index_types=Telescope.of(App(list_ind, Var(1))),  # xs : List A
        level=u,
    )

    all_nil_ctor = Ctor(
        name="all_nil",
        inductive=all_ind,
        result_indices=ArgList.of(App(nil_ctor, Var(1))),
        uarity=1,
    )

    all_cons_ctor = Ctor(
        name="all_cons",
        inductive=all_ind,
        field_schemas=Telescope.of(
            mk_app(list_ind, Var(1)),  # xs : List A
            Var(2),  # x : A
            mk_app(Var(2), Var(0)),  # px : P x
            mk_uapp(all_ind, (u,), Var(4), Var(3), Var(2)),
        ),
        result_indices=ArgList.of(mk_app(cons_ctor, Var(5), Var(2), Var(3))),  # x :: xs
        uarity=1,
    )

    object.__setattr__(all_ind, "constructors", (all_nil_ctor, all_cons_ctor))
    return all_ind, all_nil_ctor, all_cons_ctor


All_U, AllNil_U, AllCons_U = _all()


def AllAt(level: LevelExpr | int = 0) -> Term:
    return UApp(All_U, level)


def AllNilCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(AllNil_U, level)


def AllConsCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(AllCons_U, level)


All = AllAt()
AllNilCtor = AllNilCtorAt()
AllConsCtor = AllConsCtorAt()


def AllType(A: Term, P: Term, xs: Term, *, level: LevelExpr | int = 0) -> Term:
    return mk_app(AllAt(level), A, P, xs)


def AllNil(A: Term, P: Term, *, level: LevelExpr | int = 0) -> Term:
    return mk_app(AllNilCtorAt(level), A, P)


def AllCons(
    A: Term,
    P: Term,
    xs: Term,
    x: Term,
    px: Term,
    ih: Term,
    *,
    level: LevelExpr | int = 0,
) -> Term:
    return mk_app(AllConsCtorAt(level), A, P, xs, x, px, ih)


def AllElim(motive: Term, nil_case: Term, cons_case: Term, proof: Term) -> Elim:
    return Elim(
        inductive=All_U,
        motive=motive,
        cases=(nil_case, cons_case),
        scrutinee=proof,
    )
