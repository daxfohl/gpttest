"""Helpers for building generic list terms and combinators."""

from __future__ import annotations

from mltt.kernel.ast import App, Term, Univ, Var, UApp
from mltt.kernel.ind import Elim, Ctor, Ind
from mltt.kernel.levels import LVar, LevelExpr
from mltt.kernel.telescope import mk_app, mk_uapp, Telescope


def _list() -> tuple[Ind, Ctor, Ctor]:
    u = LVar(0)
    list_ind = Ind(
        name="List",
        uarity=1,
        param_types=Telescope.of(Univ(u)),
        level=u,
    )
    nil_ctor = Ctor(name="Nil", inductive=list_ind, uarity=1)
    cons_ctor = Ctor(
        name="Cons",
        inductive=list_ind,
        field_schemas=Telescope.of(
            Var(0),
            App(mk_uapp(list_ind, (u,)), Var(1)),
        ),
        uarity=1,
    )
    object.__setattr__(list_ind, "constructors", (nil_ctor, cons_ctor))
    return list_ind, nil_ctor, cons_ctor


List_U, Nil_U, Cons_U = _list()


def ListAt(level: LevelExpr | int = 0) -> Term:
    return UApp(List_U, level)


def NilCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(Nil_U, level)


def ConsCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(Cons_U, level)


List = ListAt()
NilCtor = NilCtorAt()
ConsCtor = ConsCtorAt()


def ListType(elem_ty: Term, *, level: LevelExpr | int = 0) -> Term:
    return App(ListAt(level), elem_ty)


def Nil(elem_ty: Term, *, level: LevelExpr | int = 0) -> Term:
    return App(NilCtorAt(level), elem_ty)


def Cons(elem_ty: Term, head: Term, tail: Term, *, level: LevelExpr | int = 0) -> Term:
    return mk_app(ConsCtorAt(level), elem_ty, head, tail)


def ListElim(P: Term, base: Term, step: Term, xs: Term) -> Elim:
    return Elim(
        inductive=List_U,
        motive=P,
        cases=(base, step),
        scrutinee=xs,
    )
