"""Less-or-equal inductive for natural numbers."""

from __future__ import annotations

from ..core.ast import App, Ctor, Elim, I, Lam, Term, Var
from ..core.inductive_utils import apply_term, nested_lam
from .eq import Id, Refl
from .nat import NatType, Succ
from .rtc import RTCRefl, RTCStep, RTCType

Le = I(name="Le", index_types=(NatType(), NatType()), level=0)
LeReflCtor = Ctor(
    name="le_refl",
    inductive=Le,
    arg_types=(),
    result_indices=(Var(1), Var(1)),  # Le n n
)
LeStepCtor = Ctor(
    name="le_step",
    inductive=Le,
    arg_types=(apply_term(Le, Var(1), Var(0)),),  # Le n m
    result_indices=(
        Var(2),  # n
        Succ(Var(1)),  # Succ m
    ),
)
object.__setattr__(Le, "constructors", (LeReflCtor, LeStepCtor))


def LeType(n: Term, m: Term) -> Term:
    return apply_term(Le, n, m)


def LeRefl(n: Term) -> Term:
    return apply_term(LeReflCtor, n, n)


def LeStep(n: Term, m: Term, p: Term) -> Term:
    return apply_term(LeStepCtor, n, m, p)


def LeRec(motive: Term, refl_case: Term, step_case: Term, proof: Term) -> Elim:
    """Recursor for ``Le``."""

    return Elim(
        inductive=Le, motive=motive, cases=(refl_case, step_case), scrutinee=proof
    )


PredRelation = nested_lam(
    NatType(),
    NatType(),
    body=Id(NatType(), Var(1), Succ(Var(0))),
)


def LeRTCType(n: Term, m: Term) -> Term:
    """Le expressed as the reflexive-transitive closure of the predecessor relation."""

    return RTCType(NatType(), PredRelation, m, n)


def LeRTCRefl(n: Term) -> Term:
    return RTCRefl(NatType(), PredRelation, n)


def LeRTCStep(n: Term, m: Term, ih: Term) -> Term:
    """Extend a proof of ``LeRTC n m`` to ``LeRTC n (Succ m)``."""

    step = Refl(NatType(), Succ(m))
    return RTCStep(NatType(), PredRelation, Succ(m), n, m, step, ih)


__all__ = [
    "Le",
    "LeType",
    "LeRefl",
    "LeStep",
    "LeRec",
    "LeRTCType",
    "LeRTCRefl",
    "LeRTCStep",
    "PredRelation",
]
