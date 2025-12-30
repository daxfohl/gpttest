"""Universe level expressions and comparison helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LevelNF:
    const: int
    vars: tuple[tuple[str, int], ...]

    def as_tuple(self) -> tuple[int, tuple[tuple[str, int], ...]]:
        return self.const, self.vars


class LevelExpr:
    def nf(self) -> LevelNF:
        raise NotImplementedError

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LevelExpr):
            return False
        return self.nf() == other.nf()

    def __hash__(self) -> int:
        return hash(self.nf().as_tuple())

    def __str__(self) -> str:
        return format_level(self)


@dataclass(frozen=True, eq=False)
class LevelConst(LevelExpr):
    value: int

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("Universe level must be non-negative")

    def nf(self) -> LevelNF:
        return _make_nf(self.value, {})


@dataclass(frozen=True, eq=False)
class LevelVar(LevelExpr):
    name: str

    def nf(self) -> LevelNF:
        return _make_nf(0, {self.name: 0})


@dataclass(frozen=True, eq=False)
class LevelSucc(LevelExpr):
    inner: LevelExpr

    def nf(self) -> LevelNF:
        base = self.inner.nf()
        vars_dict = {name: offset + 1 for name, offset in base.vars}
        return _make_nf(base.const + 1, vars_dict)


@dataclass(frozen=True, eq=False)
class LevelMax(LevelExpr):
    left: LevelExpr
    right: LevelExpr

    def nf(self) -> LevelNF:
        left_nf = self.left.nf()
        right_nf = self.right.nf()
        vars_dict = {name: offset for name, offset in left_nf.vars}
        for name, offset in right_nf.vars:
            vars_dict[name] = max(vars_dict.get(name, 0), offset)
        return _make_nf(max(left_nf.const, right_nf.const), vars_dict)


def _make_nf(const: int, vars_dict: dict[str, int]) -> LevelNF:
    normalized = {name: offset for name, offset in vars_dict.items() if offset >= 0}
    return LevelNF(const, tuple(sorted(normalized.items())))


def coerce_level(level: int | LevelExpr) -> LevelExpr:
    if isinstance(level, int):
        return LevelConst(level)
    return level


def level_succ(level: int | LevelExpr) -> LevelExpr:
    return LevelSucc(coerce_level(level))


def level_max(left: int | LevelExpr, right: int | LevelExpr) -> LevelExpr:
    return LevelMax(coerce_level(left), coerce_level(right))


def level_leq(left: int | LevelExpr, right: int | LevelExpr) -> bool:
    left_nf = coerce_level(left).nf()
    right_nf = coerce_level(right).nf()
    if left_nf.const > right_nf.const:
        return False
    right_vars = dict(right_nf.vars)
    for name, offset in left_nf.vars:
        if name not in right_vars:
            return False
        if offset > right_vars[name]:
            return False
    return True


def format_level(level: int | LevelExpr) -> str:
    nf = coerce_level(level).nf()
    parts: list[str] = []
    if nf.const > 0 or not nf.vars:
        parts.append(str(nf.const))
    for name, offset in nf.vars:
        if offset == 0:
            parts.append(name)
        else:
            parts.append(f"{name}+{offset}")
    if len(parts) == 1:
        return parts[0]
    return f"max({', '.join(parts)})"
