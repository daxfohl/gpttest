"""Top (unit) and Bottom (empty) inductive types."""

from __future__ import annotations

from mltt.kernel.ast import Term
from mltt.kernel.levels import LevelConst
from mltt.kernel.ind import Elim, Ctor, Ind

# Top (unit) has a single inhabitant.
Top = Ind(name="Top", level=LevelConst(0))
TtCtor = Ctor(name="tt", inductive=Top)
object.__setattr__(Top, "constructors", (TtCtor,))


def TopType() -> Ind:
    return Top


def Tt() -> Term:
    return TtCtor


def TopRec(motive: Term, case: Term, scrutinee: Term) -> Elim:
    """Eliminate Top by providing the single branch for ``tt``."""

    return Elim(inductive=Top, motive=motive, cases=(case,), scrutinee=scrutinee)


# Bottom (empty) has no constructors.
Bot = Ind(name="Bottom", level=LevelConst(0))
object.__setattr__(Bot, "constructors", ())


def BotType() -> Ind:
    return Bot


def BotRec(motive: Term, scrutinee: Term) -> Elim:
    """Ex falso eliminator for Bottom."""

    return Elim(inductive=Bot, motive=motive, cases=(), scrutinee=scrutinee)
