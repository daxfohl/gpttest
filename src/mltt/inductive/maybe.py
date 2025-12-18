"""Maybe (option) type with its constructors and recursor."""

from __future__ import annotations

from ..core.ast import App, Ctor, Elim, I, Term, Univ, Var
from ..core.util import apply_term

Maybe = I(name="Maybe", param_types=(Univ(0),), level=0)
NothingCtor = Ctor(name="Nothing", inductive=Maybe)
JustCtor = Ctor(name="Just", inductive=Maybe, arg_types=(Var(0),))
object.__setattr__(Maybe, "constructors", (NothingCtor, JustCtor))


def MaybeType(elem_ty: Term) -> Term:
    return App(Maybe, elem_ty)


def Nothing(elem_ty: Term) -> Term:
    return App(NothingCtor, elem_ty)


def Just(elem_ty: Term, value: Term) -> Term:
    return apply_term(JustCtor, elem_ty, value)


def MaybeRec(P: Term, nothing_case: Term, just_case: Term, scrutinee: Term) -> Elim:
    """Eliminate Maybe by providing branches for ``Nothing`` and ``Just``."""

    return Elim(
        inductive=Maybe,
        motive=P,
        cases=(nothing_case, just_case),
        scrutinee=scrutinee,
    )


__all__ = [
    "Maybe",
    "MaybeType",
    "NothingCtor",
    "JustCtor",
    "Nothing",
    "Just",
    "MaybeRec",
]
