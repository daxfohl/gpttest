"""Reflexive-transitive closure of a binary relation."""

from __future__ import annotations

from ..core.ast import Ctor, Elim, Ind, Pi, Term, Univ, Var
from ..core.util import apply_term

RTC = Ind(
    name="RTC",
    param_types=(
        Univ(0),  # A
        Pi(Var(0), Pi(Var(1), Univ(0))),  # R : A -> A -> Type
    ),
    index_types=(
        Var(1),  # x : A
        Var(2),  # z : A
    ),
    level=0,
)

RTCReflCtor = Ctor(
    name="rtc_refl",
    inductive=RTC,
    arg_types=(Var(1),),  # x : A
    result_indices=(
        Var(0),  # x
        Var(0),  # x
    ),
)

RTCStepCtor = Ctor(
    name="rtc_step",
    inductive=RTC,
    arg_types=(
        Var(1),  # x : A
        Var(2),  # z : A
        Var(3),  # y : A
        apply_term(Var(3), Var(2), Var(0)),  # R x y
        apply_term(RTC, Var(5), Var(4), Var(1), Var(2)),  # RTC A R y z
    ),
    result_indices=(
        Var(4),  # x
        Var(3),  # z
    ),
)

object.__setattr__(RTC, "constructors", (RTCReflCtor, RTCStepCtor))


def RTCType(A: Term, R: Term, x: Term, z: Term) -> Term:
    return apply_term(RTC, A, R, x, z)


def RTCRefl(A: Term, R: Term, x: Term) -> Term:
    return apply_term(RTCReflCtor, A, R, x)


def RTCStep(A: Term, R: Term, x: Term, z: Term, y: Term, step: Term, ih: Term) -> Term:
    return apply_term(RTCStepCtor, A, R, x, z, y, step, ih)


def RTCRec(motive: Term, refl_case: Term, step_case: Term, proof: Term) -> Elim:
    return Elim(
        inductive=RTC, motive=motive, cases=(refl_case, step_case), scrutinee=proof
    )


__all__ = ["RTC", "RTCType", "RTCRefl", "RTCStep", "RTCRec"]
