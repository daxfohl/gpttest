"""Maybe (option) type with its constructors and recursor."""

from __future__ import annotations

from functools import cache

from mltt.kernel.ast import App, Term, Univ, Var
from mltt.kernel.telescope import mk_app, Telescope
from mltt.kernel.ind import Elim, Ctor, Ind


@cache
def _maybe_family(level: int) -> tuple[Ind, Ctor, Ctor]:
    maybe_ind = Ind(name="Maybe", param_types=Telescope.of(Univ(level)), level=level)
    nothing_ctor = Ctor(name="Nothing", inductive=maybe_ind)
    just_ctor = Ctor(
        name="Just", inductive=maybe_ind, field_schemas=Telescope.of(Var(0))
    )
    object.__setattr__(maybe_ind, "constructors", (nothing_ctor, just_ctor))
    return maybe_ind, nothing_ctor, just_ctor


Maybe, NothingCtor, JustCtor = _maybe_family(0)


def MaybeAt(level: int = 0) -> Ind:
    return _maybe_family(level)[0]


def NothingCtorAt(level: int = 0) -> Ctor:
    return _maybe_family(level)[1]


def JustCtorAt(level: int = 0) -> Ctor:
    return _maybe_family(level)[2]


def MaybeType(elem_ty: Term, *, level: int = 0) -> Term:
    return App(MaybeAt(level), elem_ty)


def Nothing(elem_ty: Term, *, level: int = 0) -> Term:
    return App(NothingCtorAt(level), elem_ty)


def Just(elem_ty: Term, value: Term, *, level: int = 0) -> Term:
    return mk_app(JustCtorAt(level), elem_ty, value)


def MaybeElim(
    P: Term, nothing_case: Term, just_case: Term, scrutinee: Term, *, level: int = 0
) -> Elim:
    """Eliminate Maybe by providing branches for ``Nothing`` and ``Just``."""

    return Elim(
        inductive=MaybeAt(level),
        motive=P,
        cases=(nothing_case, just_case),
        scrutinee=scrutinee,
    )
