"""Object-oriented abstract syntax tree nodes for the miniature MLTT."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace, Field
from functools import cache
from operator import methodcaller
from typing import TYPE_CHECKING, Any, Callable, ClassVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .debruijn import Ctx, ArgList, UCtx


@dataclass(frozen=True)
class LevelExpr:
    """Base class for universe level expressions."""

    def normalize(self) -> LevelExpr:
        return self

    def as_int(self) -> int | None:
        return None

    def shift(self, by: int, cutoff: int = 0) -> LevelExpr:
        return self

    def subst(self, sub: LevelExpr, j: int = 0) -> LevelExpr:
        return self

    def instantiate(
        self, actuals: Sequence[LevelExpr], depth_above: int = 0
    ) -> LevelExpr:
        """
        Substitute ``actuals`` for the outer binder block of ``self``.

        This mirrors ``Term.instantiate`` but operates over level variables.
        """
        level = self
        k = len(actuals)
        for i, actual in enumerate(actuals):
            index = depth_above + k - i - 1
            level = level.subst(actual.shift(index), index)
        return level

    def check(self, uctx: UCtx) -> None:
        return None


@dataclass(frozen=True)
class ConstLevel(LevelExpr):
    """A concrete universe level."""

    value: int = 0

    def __post_init__(self) -> None:
        if self.value < 0:
            raise ValueError("Universe level must be non-negative")

    def as_int(self) -> int | None:
        return self.value


@dataclass(frozen=True)
class SuccLevel(LevelExpr):
    """A successor universe level."""

    pred: LevelExpr

    def normalize(self) -> LevelExpr:
        pred = normalize_level(self.pred)
        pred_int = pred.as_int()
        if pred_int is not None:
            return ConstLevel(pred_int + 1)
        if pred is not self.pred:
            return SuccLevel(pred)
        return self

    def shift(self, by: int, cutoff: int = 0) -> LevelExpr:
        pred = self.pred.shift(by, cutoff)
        if pred is not self.pred:
            return SuccLevel(pred)
        return self

    def subst(self, sub: LevelExpr, j: int = 0) -> LevelExpr:
        pred = self.pred.subst(sub, j)
        if pred is not self.pred:
            return SuccLevel(pred)
        return self

    def check(self, uctx: UCtx) -> None:
        self.pred.check(uctx)


@dataclass(frozen=True)
class MaxOfLevels(LevelExpr):
    """The maximum of one or more universe levels."""

    levels: tuple[LevelExpr, ...]

    def normalize(self) -> LevelExpr:
        flattened: list[LevelExpr] = []
        for level in self.levels:
            norm = normalize_level(level)
            if isinstance(norm, MaxOfLevels):
                flattened.extend(norm.levels)
            else:
                flattened.append(norm)
        if not flattened:
            return ConstLevel(0)
        const_levels = [level for level in flattened if isinstance(level, ConstLevel)]
        if len(const_levels) == len(flattened):
            return ConstLevel(max(level.value for level in const_levels))
        unique: list[LevelExpr] = []
        for level in flattened:
            if level not in unique:
                unique.append(level)
        if len(unique) == 1:
            return unique[0]
        return MaxOfLevels(tuple(unique))

    def shift(self, by: int, cutoff: int = 0) -> LevelExpr:
        shifted = tuple(level.shift(by, cutoff) for level in self.levels)
        if shifted != self.levels:
            return MaxOfLevels(shifted)
        return self

    def subst(self, sub: LevelExpr, j: int = 0) -> LevelExpr:
        substituted = tuple(level.subst(sub, j) for level in self.levels)
        if substituted != self.levels:
            return MaxOfLevels(substituted)
        return self

    def check(self, uctx: UCtx) -> None:
        for level in self.levels:
            level.check(uctx)


@dataclass(frozen=True)
class LevelVar(LevelExpr):
    """A de Bruijn universe level variable."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("Universe variable indices must be non-negative")

    def shift(self, by: int, cutoff: int = 0) -> LevelExpr:
        if self.k >= cutoff:
            return LevelVar(self.k + by)
        return self

    def subst(self, sub: LevelExpr, j: int = 0) -> LevelExpr:
        if self.k == j:
            return sub
        if self.k > j:
            return LevelVar(self.k - 1)
        return self

    def check(self, uctx: UCtx) -> None:
        if self.k >= len(uctx):
            raise TypeError(f"Unbound universe variable {self.k}")


LevelLike = LevelExpr | int


def normalize_level(level: LevelLike) -> LevelExpr:
    if isinstance(level, int):
        return ConstLevel(level)
    if isinstance(level, LevelExpr):
        return level.normalize()
    raise TypeError(f"Unexpected level expression: {level!r}")


def max_level(*levels: LevelLike) -> LevelExpr:
    normalized = [normalize_level(level) for level in levels]
    return normalize_level(MaxOfLevels(tuple(normalized)))


def succ_level(level: LevelLike) -> LevelExpr:
    return normalize_level(SuccLevel(normalize_level(level)))


def _level_int(level: LevelLike) -> int | None:
    return normalize_level(level).as_int()


def level_lt(left: LevelLike, right: LevelLike) -> bool:
    left_int = _level_int(left)
    right_int = _level_int(right)
    if left_int is None or right_int is None:
        raise TypeError(
            "Cannot compare non-constant universe levels:\n"
            f"  left = {left}\n"
            f"  right = {right}"
        )
    return left_int < right_int


def level_leq(left: LevelLike, right: LevelLike) -> bool:
    left_int = _level_int(left)
    right_int = _level_int(right)
    if left_int is None or right_int is None:
        raise TypeError(
            "Cannot compare non-constant universe levels:\n"
            f"  left = {left}\n"
            f"  right = {right}"
        )
    return left_int <= right_int


def _map_term_values(value: Any, f: Callable[[Term], Term]) -> Any:
    if isinstance(value, Term):
        return f(value)
    if isinstance(value, tuple):
        return tuple(_map_term_values(v, f) for v in value)
    return value


def _map_level_values(
    value: Any,
    term_mapper: Callable[[Term], Term],
    level_mapper: Callable[[LevelExpr], LevelExpr],
    *,
    allow_term_recursion: bool,
) -> Any:
    if isinstance(value, LevelExpr):
        return level_mapper(value)
    if isinstance(value, Term):
        if allow_term_recursion:
            return term_mapper(value)
        return value
    if isinstance(value, tuple):
        return tuple(
            _map_level_values(
                v,
                term_mapper,
                level_mapper,
                allow_term_recursion=allow_term_recursion,
            )
            for v in value
        )
    return value


@dataclass(frozen=True, kw_only=True)
class TermFieldMeta:
    binder_count: int = 0
    unchecked: bool = False


def meta(f: Field) -> TermFieldMeta:
    return f.metadata.get("") or TermFieldMeta()


@dataclass(frozen=True)
class Term:
    """Base class for all MLTT terms."""

    is_terminal: ClassVar[bool] = False

    @classmethod
    @cache
    def _reducible_fields(cls) -> tuple[Field, ...]:
        return () if cls.is_terminal else fields(cls)

    @classmethod
    @cache
    def _checkable_fields(cls) -> tuple[Field, ...]:
        return tuple(f for f in fields(cls) if not meta(f).unchecked)

    def _map_field(
        self, f: Field, mapper: Callable[[Term, TermFieldMeta], Term]
    ) -> Any:
        value = getattr(self, f.name)
        return _map_term_values(value, lambda t: mapper(t, meta(f)))

    def _replace_terms(self, mapper: Callable[[Term, TermFieldMeta], Term]) -> Term:
        updates = {f.name: self._map_field(f, mapper) for f in self._reducible_fields()}
        # noinspection PyArgumentList
        return replace(self, **updates) if updates else self

    def _replace_levels(self, mapper: Callable[[LevelExpr], LevelExpr]) -> Term:
        def term_mapper(t: Term) -> Term:
            return t._replace_levels(mapper)

        updates = {}
        for f in fields(self):
            value = getattr(self, f.name)
            new_value = _map_level_values(
                value,
                term_mapper,
                mapper,
                allow_term_recursion=not self.is_terminal,
            )
            if new_value != value:
                updates[f.name] = new_value
        # noinspection PyArgumentList
        return replace(self, **updates) if updates else self

    def shift(self, by: int, cutoff: int = 0) -> Term:
        """Shift free variables in the term."""
        return self._replace_terms(lambda t, m: t.shift(by, cutoff + m.binder_count))

    def subst(self, sub: Term, j: int = 0) -> Term:
        """Substitute ``sub`` for ``Var(j)`` inside the term."""
        return self._replace_terms(
            lambda t, m: t.subst(sub.shift(m.binder_count), j + m.binder_count)
        )

    def instantiate(self, actuals: ArgList, depth_above: int = 0) -> Term:
        """
        Substitute ``actuals`` for the outer binder block of ``self``.

        Self is assumed written under (actuals)(...) where ``depth_above`` is the number
        of binders *below* the actuals block that remain in scope at substitution time.
        For each actual, eliminate at de Bruijn index:
            index = depth_above + len(actuals) - i - 1
        using the project’s convention:
            schema = schema.subst(actual.shift(index), index)
        """
        t = self
        k = len(actuals)
        for i, a in enumerate(actuals):
            index = depth_above + k - i - 1
            t = t.subst(a.shift(index), index)
        return t

    def level_shift(self, by: int, cutoff: int = 0) -> Term:
        """Shift free universe variables in the term."""
        return self._replace_levels(lambda level: level.shift(by, cutoff))

    def level_subst(self, sub: LevelExpr, j: int = 0) -> Term:
        """Substitute ``sub`` for ``LevelVar(j)`` inside the term."""
        return self._replace_levels(lambda level: level.subst(sub, j))

    def level_instantiate(
        self, actuals: Sequence[LevelExpr], depth_above: int = 0
    ) -> Term:
        """
        Substitute ``actuals`` for the outer universe binder block of ``self``.

        Mirrors ``instantiate`` but operates on universe level variables.
        """
        term = self
        k = len(actuals)
        for i, actual in enumerate(actuals):
            index = depth_above + k - i - 1
            term = term.level_subst(actual.shift(index), index)
        return term

    # --- Reduction ------------------------------------------------------------
    def whnf_step(self) -> Term:
        """Weak head normal form."""
        return self

    def whnf(self) -> Term:
        """Weak head normal form."""
        reduced = self.whnf_step()
        if reduced != self:
            return reduced.whnf()
        return self

    def _reduce_inside_step(self, reducer: Callable[[Term], Term]) -> Term:
        reduced = reducer(self)
        if reduced != self:
            return reduced
        for f in type(self)._reducible_fields():
            curr = getattr(self, f.name)
            new = _map_term_values(curr, lambda t: t._reduce_inside_step(reducer))
            if new != curr:
                # noinspection PyArgumentList
                return replace(self, **{f.name: new})
        return self

    def normalize_step(self) -> Term:
        """Perform a single beta/iota reduction step anywhere in the term."""
        return self._reduce_inside_step(methodcaller("whnf_step"))

    def normalize(self) -> Term:
        """Fully normalize the term."""
        reduced = self.normalize_step()
        if reduced != self:
            return reduced.normalize()
        return self

    # --- Typing ---------------------------------------------------------------
    def infer_type(self, ctx: Ctx | None = None, uctx: UCtx | None = None) -> Term:
        from .debruijn import Ctx, UCtx

        return self._infer_type(ctx or Ctx(), uctx or UCtx())

    def _infer_type(self, ctx: Ctx, uctx: UCtx) -> Term:
        raise TypeError(f"Unexpected term in infer_type:\n  term = {self!r}")

    def type_check(
        self, ty: Term, ctx: Ctx | None = None, uctx: UCtx | None = None
    ) -> None:
        from .debruijn import Ctx, UCtx

        self._type_check(ty.whnf(), ctx or Ctx(), uctx or UCtx())

    def _type_check(self, expected_ty: Term, ctx: Ctx, uctx: UCtx) -> None:
        self._check_against_inferred(expected_ty, ctx, uctx)

    def expect_universe(
        self, ctx: Ctx | None = None, uctx: UCtx | None = None
    ) -> LevelExpr:
        ty = self.infer_type(ctx, uctx).whnf()
        if not isinstance(ty, Univ):
            raise TypeError(
                "Expected a universe:\n" f"  term = {self}\n" f"  inferred = {ty}"
            )
        return normalize_level(ty.level)

    def type_equal(self, other: Term) -> bool:
        self_whnf = self.whnf()
        other_whnf = other.whnf()
        if self_whnf == other_whnf:
            return True
        if type(self_whnf) is not type(other_whnf):
            return False
        for f in type(self_whnf)._checkable_fields():
            s = getattr(self_whnf, f.name)
            o = getattr(other_whnf, f.name)
            if isinstance(s, Term) and isinstance(o, Term):
                if not s.type_equal(o):
                    return False
            elif s != o:
                return False
        return True

    def _check_against_inferred(self, expected_ty: Term, ctx: Ctx, uctx: UCtx) -> None:
        inferred_ty = self.infer_type(ctx, uctx)
        if not expected_ty.type_equal(inferred_ty):
            raise TypeError(
                f"{type(self).__name__} type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}\n"
                f"  inferred = {inferred_ty}"
            )

    # --- Display --------------------------------------------------------------
    def __str__(self) -> str:
        # Deferred import avoids cycles when pretty-printing dataclass reprs.
        from .pretty import pretty

        return pretty(self)


@dataclass(frozen=True)
class Var(Term):
    """De Bruijn variable pointing to the binder at ``k``."""

    k: int
    is_terminal: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("De Bruijn indices must be non-negative")

    # De Bruijn ---------------------------------------------------------------
    def shift(self, by: int, cutoff: int = 0) -> Term:
        return Var(self.k + by if self.k >= cutoff else self.k)

    def subst(self, sub: Term, j: int = 0) -> Term:
        if self.k == j:
            return sub
        if self.k > j:
            return Var(self.k - 1)
        return self

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx, uctx: UCtx) -> Term:
        if self.k < len(ctx):
            return ctx[self.k].ty.shift(self.k + 1)
        raise TypeError(f"Unbound variable {self.k}")

    def _type_check(self, expected_ty: Term, ctx: Ctx, uctx: UCtx) -> None:
        if self.k >= len(ctx):
            raise TypeError(f"Unbound variable {self.k}")
        found_ty = ctx[self.k].ty.shift(self.k + 1)
        if not found_ty.type_equal(expected_ty):
            raise TypeError(
                "Variable type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}\n"
                f"  found = {found_ty}"
            )


@dataclass(frozen=True)
class Lam(Term):
    """Dependent lambda term with an argument type and body."""

    arg_ty: Term
    body: Term = field(metadata={"": TermFieldMeta(binder_count=1)})

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx, uctx: UCtx) -> Term:
        body_ty = self.body.infer_type(ctx.push(self.arg_ty), uctx)
        return Pi(self.arg_ty, body_ty)

    def _type_check(self, expected_ty: Term, ctx: Ctx, uctx: UCtx) -> None:
        if not isinstance(expected_ty, Pi):
            raise TypeError(
                "Lambda expected to have Pi type:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}"
            )
        if not self.arg_ty.type_equal(expected_ty.arg_ty):
            raise TypeError(
                "Lambda domain mismatch:\n"
                f"  term = {self}\n"
                f"  expected domain = {expected_ty.arg_ty}\n"
                f"  found domain = {self.arg_ty}"
            )
        self.body.type_check(expected_ty.return_ty, ctx.push(self.arg_ty), uctx)


@dataclass(frozen=True)
class Pi(Term):
    """Dependent function type (Pi-type)."""

    arg_ty: Term
    return_ty: Term = field(metadata={"": TermFieldMeta(binder_count=1)})

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx, uctx: UCtx) -> Term:
        arg_level = self.arg_ty.expect_universe(ctx, uctx)
        body_level = self.return_ty.expect_universe(ctx.push(self.arg_ty), uctx)
        return Univ(max_level(arg_level, body_level))


@dataclass(frozen=True)
class App(Term):
    """Function application."""

    func: Term
    arg: Term

    # Reduction ----------------------------------------------------------------
    def whnf_step(self) -> Term:
        f_whnf = self.func.whnf()
        if isinstance(f_whnf, Lam):
            return f_whnf.body.subst(self.arg)
        return App(f_whnf, self.arg)

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx, uctx: UCtx) -> Term:
        f_ty = self.func.infer_type(ctx, uctx).whnf()
        if not isinstance(f_ty, Pi):
            raise TypeError(
                "Application of non-function:\n"
                f"  term = {self}\n"
                f"  function = {self.func}\n"
                f"  inferred f_ty = {f_ty}"
            )
        self.arg.type_check(f_ty.arg_ty, ctx, uctx)
        return f_ty.return_ty.subst(self.arg)

    def _type_check(self, expected_ty: Term, ctx: Ctx, uctx: UCtx) -> None:
        f_ty = self.func.infer_type(ctx, uctx).whnf()
        if not isinstance(f_ty, Pi):
            raise TypeError(
                "Application of non-function:\n"
                f"  term = {self}\n"
                f"  function = {self.func}\n"
                f"  inferred f_ty = {f_ty}"
            )
        self.arg.type_check(f_ty.arg_ty, ctx, uctx)
        inferred_ty = f_ty.return_ty.subst(self.arg)
        if not expected_ty.type_equal(inferred_ty):
            raise TypeError(
                "Application result type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}\n"
                f"  inferred = {inferred_ty}"
            )


@dataclass(frozen=True)
class Univ(Term):
    """A universe ``Type(level)``."""

    level: LevelLike = 0
    is_terminal: ClassVar[bool] = True

    def __post_init__(self) -> None:
        normalized = normalize_level(self.level)
        object.__setattr__(self, "level", normalized)

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx, uctx: UCtx) -> Term:
        level = normalize_level(self.level)
        level.check(uctx)
        return Univ(succ_level(level))

    def _type_check(self, expected_ty: Term, ctx: Ctx, uctx: UCtx) -> None:
        if not isinstance(expected_ty, Univ):
            raise TypeError(
                "Universe type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}"
            )
        found_level = normalize_level(self.level)
        found_level.check(uctx)
        expected_level = normalize_level(expected_ty.level)
        expected_level.check(uctx)
        if not level_leq(succ_level(found_level), expected_level):
            raise TypeError(
                "Universe level mismatch:\n"
                f"  term = {self}\n"
                f"  expected level = {expected_level}\n"
                f"  found level = {found_level}"
            )


__all__ = [
    "ConstLevel",
    "LevelExpr",
    "LevelVar",
    "MaxOfLevels",
    "SuccLevel",
    "Term",
    "Var",
    "Lam",
    "Pi",
    "App",
    "Univ",
]
