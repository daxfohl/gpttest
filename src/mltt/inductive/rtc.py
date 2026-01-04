"""Reflexive-transitive closure of a binary relation."""

from __future__ import annotations

from mltt.kernel.ast import Pi, Term, Univ, Var
from mltt.kernel.ind import Elim, Ctor, Ind
from mltt.kernel.tel import mk_app, Telescope, Spine

RTC = Ind(
    name="RTC",
    param_types=Telescope.of(
        Univ(0),  # A
        Pi(Var(0), Pi(Var(1), Univ(0))),  # R : A -> A -> Type
    ),
    index_types=Telescope.of(
        Var(1),  # x : A
        Var(2),  # z : A
    ),
    level=0,
)

RTCReflCtor = Ctor(
    name="rtc_refl",
    inductive=RTC,
    field_schemas=Telescope.of(Var(1)),  # x : A
    result_indices=Spine.of(
        Var(0),  # x
        Var(0),  # x
    ),
)

RTCStepCtor = Ctor(
    name="rtc_step",
    inductive=RTC,
    field_schemas=Telescope.of(
        Var(1),  # x : A
        Var(2),  # z : A
        Var(3),  # y : A
        mk_app(Var(3), Var(2), Var(0)),  # R x y
        mk_app(RTC, Var(5), Var(4), Var(1), Var(2)),  # RTC A R y z
    ),
    result_indices=Spine.of(
        Var(4),  # x
        Var(3),  # z
    ),
)

object.__setattr__(RTC, "constructors", (RTCReflCtor, RTCStepCtor))


def RTCType(A: Term, R: Term, x: Term, z: Term) -> Term:
    return mk_app(RTC, A, R, x, z)


def RTCRefl(A: Term, R: Term, x: Term) -> Term:
    return mk_app(RTCReflCtor, A, R, x)


def RTCStep(A: Term, R: Term, x: Term, z: Term, y: Term, step: Term, ih: Term) -> Term:
    return mk_app(RTCStepCtor, A, R, x, z, y, step, ih)


def RTCElim(motive: Term, refl_case: Term, step_case: Term, proof: Term) -> Elim:
    return Elim(
        inductive=RTC, motive=motive, cases=(refl_case, step_case), scrutinee=proof
    )
