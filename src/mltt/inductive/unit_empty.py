"""Unit and Empty inductive types."""

from __future__ import annotations

from mltt.kernel.ast import Term
from mltt.kernel.ind import Elim, Ctor, Ind

# Unit has a single inhabitant.
Unit = Ind(name="Unit", level=0)
UnitCtor = Ctor(name="tt", inductive=Unit)
object.__setattr__(Unit, "constructors", (UnitCtor,))


def UnitType() -> Ind:
    return Unit


def UnitValue() -> Term:
    return UnitCtor


def UnitElim(motive: Term, case: Term, scrutinee: Term) -> Elim:
    """Eliminate Unit by providing the single branch for ``tt``."""

    return Elim(inductive=Unit, motive=motive, cases=(case,), scrutinee=scrutinee)


# Empty has no constructors.
Empty = Ind(name="Empty", level=0)
object.__setattr__(Empty, "constructors", ())


def EmptyType() -> Ind:
    return Empty


def EmptyElim(motive: Term, scrutinee: Term) -> Elim:
    """Ex falso eliminator for Empty."""

    return Elim(inductive=Empty, motive=motive, cases=(), scrutinee=scrutinee)
