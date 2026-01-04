"""Universe level metadata and constraints for the solver."""

from __future__ import annotations

from dataclasses import dataclass, field

from mltt.common.span import Span
from mltt.kernel.levels import LConst, LevelExpr


@dataclass
class LMetaInfo:
    solution: LevelExpr | None = None
    span: Span | None = None
    origin: str = "type"
    lower_bound: LevelExpr = field(default_factory=lambda: LConst(0))
    upper_bounds: list[LevelExpr] = field(default_factory=list)


@dataclass
class LevelConstraint:
    lhs: LevelExpr
    rhs: LevelExpr
    span: Span | None = None
    reason: str | None = None
