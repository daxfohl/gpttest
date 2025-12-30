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

    def eval(self) -> int | None:
        match self:
            case LConst(k):
                return k
            case LSucc(e):
                inner = e.eval()
                if inner is None:
                    return None
                return inner + 1
            case LMax(a, b):
                left = a.eval()
                right = b.eval()
                if left is None or right is None:
                    return None
                return max(left, right)
            case LVar():
                return None
            case _:
                return None

    def succ(self) -> LevelExpr:
        value = self.eval()
        if value is not None:
            return LConst(value + 1)
        return LSucc(self)

    def max(self, other: LevelExpr) -> LevelExpr:
        a_value = self.eval()
        b_value = other.eval()
        if a_value is not None and b_value is not None:
            return LConst(max(a_value, b_value))
        if self == other:
            return self
        return LMax(self, other)

    def __ge__(self, other: LevelExpr) -> bool:
        if self == other:
            return True
        a_eval = self.eval()
        b_eval = other.eval()
        if a_eval is not None and b_eval is not None:
            return a_eval >= b_eval
        match (self, other):
            case (LMax(a1, a2), _):
                return a1 >= other or a2 >= other
            case (_, LMax(b1, b2)):
                return self >= b1 and self >= b2
            case (LSucc(a1), LSucc(b1)):
                return a1 >= b1
            case (LSucc(a1), _):
                return a1 >= other
        if a_eval is None or b_eval is None:
            return True
        return False

    def __str__(self) -> str:
        match self:
            case LConst(k):
                return str(k)
            case LVar(k):
                return f"u{k}"
            case LSucc(e):
                return f"{e}+1"
            case LMax(a, b):
                return f"max({a}, {b})"
        return repr(self)


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
