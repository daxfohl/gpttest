"""Helpers for building generic list terms and combinators."""

from __future__ import annotations

from functools import cache

from ..core.ast import App, ConstLevel, LevelExpr, LevelLike, Term, Univ, Var
from ..core.debruijn import mk_app, Telescope
from ..core.ind import Elim, Ctor, Ind


@cache
def _list_family_norm(level: LevelExpr) -> tuple[Ind, Ctor, Ctor]:
    list_ind = Ind(name="List", param_types=Telescope.of(Univ(level)), level=level)
    nil_ctor = Ctor(name="Nil", inductive=list_ind)
    cons_ctor = Ctor(
        name="Cons",
        inductive=list_ind,
        field_schemas=Telescope.of(
            Var(0),
            App(list_ind, Var(1)),
        ),
    )
    object.__setattr__(list_ind, "constructors", (nil_ctor, cons_ctor))
    return list_ind, nil_ctor, cons_ctor


def _normalize_level(level: LevelLike) -> LevelExpr:
    if isinstance(level, LevelExpr):
        return level
    return ConstLevel(level)


def _list_family(level: LevelLike) -> tuple[Ind, Ctor, Ctor]:
    return _list_family_norm(_normalize_level(level))


List, NilCtor, ConsCtor = _list_family(0)


def ListAt(level: LevelLike) -> Ind:
    return _list_family(level)[0]


def NilCtorAt(level: LevelLike) -> Ctor:
    return _list_family(level)[1]


def ConsCtorAt(level: LevelLike) -> Ctor:
    return _list_family(level)[2]


def ListType(elem_ty: Term, *, level: LevelLike = 0) -> Term:
    return App(ListAt(level), elem_ty)


def Nil(elem_ty: Term, *, level: LevelLike = 0) -> Term:
    return App(NilCtorAt(level), elem_ty)


def Cons(elem_ty: Term, head: Term, tail: Term, *, level: LevelLike = 0) -> Term:
    return mk_app(ConsCtorAt(level), elem_ty, head, tail)


def ListElim(
    P: Term, base: Term, step: Term, xs: Term, *, level: LevelLike = 0
) -> Elim:
    inductive = ListAt(level)
    return Elim(
        inductive=inductive,
        motive=P,
        cases=(base, step),
        scrutinee=xs,
    )
