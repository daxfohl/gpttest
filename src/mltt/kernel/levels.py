"""Universe level expressions and operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LevelExpr:
    """Universe level expressions with de Bruijn indices."""

    @staticmethod
    def of(level: LevelExpr | int) -> LevelExpr:
        return LConst(level) if isinstance(level, int) else level

    def shift(self, by: int, cutoff: int = 0) -> LevelExpr:
        return self

    def subst(self, sub: LevelExpr, j: int = 0) -> LevelExpr:
        return self

    def instantiate(
        self, actuals: tuple[LevelExpr, ...], depth_above: int = 0
    ) -> LevelExpr:
        e: LevelExpr = self
        k = len(actuals)
        for i, a in enumerate(actuals):
            index = depth_above + k - i - 1
            e = e.subst(a.shift(index), index)
        return e


@dataclass(frozen=True)
class LConst(LevelExpr):
    k: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("Universe levels must be non-negative")


@dataclass(frozen=True)
class LVar(LevelExpr):
    """De Bruijn level variable (0 = innermost u-binder)."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("De Bruijn indices must be non-negative")

    def shift(self, by: int, cutoff: int = 0) -> LevelExpr:
        return LVar(self.k + by if self.k >= cutoff else self.k)

    def subst(self, sub: LevelExpr, j: int = 0) -> LevelExpr:
        if self.k == j:
            return sub
        if self.k > j:
            return LVar(self.k - 1)
        return self


@dataclass(frozen=True)
class LSucc(LevelExpr):
    e: LevelExpr

    def shift(self, by: int, cutoff: int = 0) -> LevelExpr:
        return LSucc(self.e.shift(by, cutoff))

    def subst(self, sub: LevelExpr, j: int = 0) -> LevelExpr:
        return LSucc(self.e.subst(sub, j))


@dataclass(frozen=True)
class LMax(LevelExpr):
    a: LevelExpr
    b: LevelExpr

    def shift(self, by: int, cutoff: int = 0) -> LevelExpr:
        return LMax(self.a.shift(by, cutoff), self.b.shift(by, cutoff))

    def subst(self, sub: LevelExpr, j: int = 0) -> LevelExpr:
        return LMax(self.a.subst(sub, j), self.b.subst(sub, j))


def _level_eval(level: LevelExpr) -> int | None:
    match level:
        case LConst(k):
            return k
        case LSucc(e):
            inner = _level_eval(e)
            if inner is None:
                return None
            return inner + 1
        case LMax(a, b):
            left = _level_eval(a)
            right = _level_eval(b)
            if left is None or right is None:
                return None
            return max(left, right)
        case LVar():
            return None
        case _:
            return None


def _level_succ(level: LevelExpr) -> LevelExpr:
    value = _level_eval(level)
    if value is not None:
        return LConst(value + 1)
    return LSucc(level)


def _level_max(a: LevelExpr, b: LevelExpr) -> LevelExpr:
    a_value = _level_eval(a)
    b_value = _level_eval(b)
    if a_value is not None and b_value is not None:
        return LConst(max(a_value, b_value))
    if a == b:
        return a
    return LMax(a, b)


def _level_geq(a: LevelExpr, b: LevelExpr) -> bool:
    if a == b:
        return True
    a_eval = _level_eval(a)
    b_eval = _level_eval(b)
    if a_eval is not None and b_eval is not None:
        return a_eval >= b_eval
    match (a, b):
        case (LMax(a1, a2), _):
            return _level_geq(a1, b) or _level_geq(a2, b)
        case (_, LMax(b1, b2)):
            return _level_geq(a, b1) and _level_geq(a, b2)
        case (LSucc(a1), LSucc(b1)):
            return _level_geq(a1, b1)
        case (LSucc(a1), _):
            return _level_geq(a1, b)
    if a_eval is None or b_eval is None:
        return True
    return False


def format_level(level: LevelExpr) -> str:
    match level:
        case LConst(k):
            return str(k)
        case LVar(k):
            return f"u{k}"
        case LSucc(e):
            return f"{format_level(e)}+1"
        case LMax(a, b):
            return f"max({format_level(a)}, {format_level(b)})"
    return repr(level)
