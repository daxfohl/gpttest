"""Maybe (option) type with its constructors and recursor."""

from __future__ import annotations

from mltt.kernel.ast import App, Term, Univ, Var, UApp
from mltt.kernel.ind import Elim, Ctor, Ind
from mltt.kernel.levels import LVar, LevelExpr
from mltt.kernel.tel import mk_app, Telescope


def _maybe() -> tuple[Ind, Ctor, Ctor]:
    u = LVar(0)
    maybe_ind = Ind(
        name="Maybe",
        uarity=1,
        param_types=Telescope.of(Univ(u)),
        level=u,
    )
    nothing_ctor = Ctor(name="Nothing", inductive=maybe_ind, uarity=1)
    just_ctor = Ctor(
        name="Just",
        inductive=maybe_ind,
        field_schemas=Telescope.of(Var(0)),
        uarity=1,
    )
    object.__setattr__(maybe_ind, "constructors", (nothing_ctor, just_ctor))
    return maybe_ind, nothing_ctor, just_ctor


Maybe_U, Nothing_U, Just_U = _maybe()


def MaybeAt(level: LevelExpr | int = 0) -> Term:
    return UApp(Maybe_U, level)


def NothingCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(Nothing_U, level)


def JustCtorAt(level: LevelExpr | int = 0) -> Term:
    return UApp(Just_U, level)


Maybe = MaybeAt()
NothingCtor = NothingCtorAt()
JustCtor = JustCtorAt()


def MaybeType(elem_ty: Term, *, level: LevelExpr | int = 0) -> Term:
    return App(MaybeAt(level), elem_ty)


def Nothing(elem_ty: Term, *, level: LevelExpr | int = 0) -> Term:
    return App(NothingCtorAt(level), elem_ty)


def Just(elem_ty: Term, value: Term, *, level: LevelExpr | int = 0) -> Term:
    return mk_app(JustCtorAt(level), elem_ty, value)


def MaybeElim(P: Term, nothing_case: Term, just_case: Term, scrutinee: Term) -> Elim:
    """Eliminate Maybe by providing branches for ``Nothing`` and ``Just``."""

    return Elim(
        inductive=Maybe_U,
        motive=P,
        cases=(nothing_case, just_case),
        scrutinee=scrutinee,
    )
