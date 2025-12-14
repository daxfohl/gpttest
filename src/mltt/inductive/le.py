"""Less-or-equal inductive for natural numbers."""

from __future__ import annotations

from ..core.ast import App, Ctor, Elim, I, Term, Var
from ..core.inductive_utils import apply_term
from .nat import NatType, Succ

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


__all__ = ["Le", "LeType", "LeRefl", "LeStep", "LeRec"]
