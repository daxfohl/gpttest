"""Less-or-equal inductive for natural numbers."""

from __future__ import annotations

from mltt.inductive.eq import Id, Refl
from mltt.inductive.nat import NatType, Succ
from mltt.inductive.rtc import RTCRefl, RTCStep, RTCType
from mltt.core.ast import Term, Var
from mltt.core.debruijn import mk_app, mk_lams, Telescope, ArgList
from mltt.core.ind import Elim, Ctor, Ind

Le = Ind(name="Le", index_types=Telescope.of(NatType(), NatType()), level=0)
LeReflCtor = Ctor(
    name="le_refl",
    inductive=Le,
    field_schemas=Telescope.of(NatType()),  # n : Nat
    result_indices=ArgList.of(Var(0), Var(0)),  # Le n n
)
LeStepCtor = Ctor(
    name="le_step",
    inductive=Le,
    field_schemas=Telescope.of(
        NatType(),  # n : Nat
        NatType(),  # m : Nat
        mk_app(Le, Var(1), Var(0)),  # Le n m
    ),
    result_indices=ArgList.of(
        Var(2),  # n
        Succ(Var(1)),  # Succ m
    ),
)
object.__setattr__(Le, "constructors", (LeReflCtor, LeStepCtor))


def LeType(n: Term, m: Term) -> Term:
    return mk_app(Le, n, m)


def LeRefl(n: Term) -> Term:
    return mk_app(LeReflCtor, n)


def LeStep(n: Term, m: Term, p: Term) -> Term:
    return mk_app(LeStepCtor, n, m, p)


def LeRec(motive: Term, refl_case: Term, step_case: Term, proof: Term) -> Elim:
    """Recursor for ``Le``."""

    return Elim(
        inductive=Le, motive=motive, cases=(refl_case, step_case), scrutinee=proof
    )


PredRelation = mk_lams(
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
