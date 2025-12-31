"""Object-oriented abstract syntax tree nodes for the miniature MLTT."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace, Field
from functools import cache
from operator import methodcaller
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from mltt.kernel.levels import LevelExpr, LConst

if TYPE_CHECKING:
    from mltt.kernel.telescope import ArgList
    from mltt.kernel.env import Env


def _map_term_values(value: Any, f: Callable[[Term], Term]) -> Any:
    if isinstance(value, Term):
        return f(value)
    if isinstance(value, tuple):
        return tuple(_map_term_values(v, f) for v in value)
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

    def inst_levels(self, actuals: tuple[LevelExpr, ...]) -> Term:
        """Instantiate universe levels in the term."""
        return self._replace_terms(lambda t, _m: t.inst_levels(actuals))

    # --- Reduction ------------------------------------------------------------
    def _whnf_step(self, env: Env) -> Term:
        """Weak head normal form."""
        return self

    def whnf_step(self, env: Env | None = None) -> Term:
        """Weak head normal form."""
        from mltt.kernel.env import Env

        return self._whnf_step(env or Env())

    def whnf(self, env: Env | None = None) -> Term:
        """Weak head normal form."""
        from mltt.kernel.env import Env

        env = env or Env()
        reduced = self.whnf_step(env)
        if reduced != self:
            return reduced.whnf(env)
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
    def infer_type(self, env: Env | None = None) -> Term:
        from mltt.kernel.env import Env

        return self._infer_type(env or Env())

    def _infer_type(self, env: Env) -> Term:
        raise TypeError(f"Unexpected term in infer_type:\n  term = {self!r}")

    def type_check(self, ty: Term, env: Env | None = None) -> None:
        from mltt.kernel.env import Env

        env = env or Env()
        self._type_check(ty.whnf(env), env)

    def _type_check(self, expected_ty: Term, env: Env) -> None:
        self._check_against_inferred(expected_ty, env)

    def expect_universe(self, env: Env | None = None) -> LevelExpr:
        ty = self.infer_type(env).whnf(env)
        if not isinstance(ty, Univ):
            raise TypeError(
                "Expected a universe:\n" f"  term = {self}\n" f"  inferred = {ty}"
            )
        return ty.level

    def _type_equal(self, other: Term, env: Env) -> bool:
        self_whnf = self.whnf(env)
        other_whnf = other.whnf(env)
        if self_whnf == other_whnf:
            return True
        if type(self_whnf) is not type(other_whnf):
            return False
        for f in type(self_whnf)._checkable_fields():
            s = getattr(self_whnf, f.name)
            o = getattr(other_whnf, f.name)
            if isinstance(s, Term) and isinstance(o, Term):
                if not s._type_equal(o, env):
                    return False
            elif s != o:
                return False
        return True

    def type_equal(self, other: Term, env: Env | None = None) -> bool:
        from mltt.kernel.env import Env

        env = env or Env()
        return self._type_equal(other, env)

    def _check_against_inferred(self, expected_ty: Term, env: Env) -> None:
        inferred_ty = self.infer_type(env)
        if not expected_ty._type_equal(inferred_ty, env):
            raise TypeError(
                f"{type(self).__name__} type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}\n"
                f"  inferred = {inferred_ty}"
            )

    # --- Display --------------------------------------------------------------
    def __str__(self) -> str:
        # Deferred import avoids cycles when pretty-printing dataclass reprs.
        from mltt.kernel.pretty import pretty

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
    def _infer_type(self, env: Env) -> Term:
        return env.local_type(self.k)

    def _type_check(self, expected_ty: Term, env: Env) -> None:
        found_ty = self._infer_type(env)
        if not found_ty._type_equal(expected_ty, env):
            raise TypeError(
                "Variable type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}\n"
                f"  found = {found_ty}"
            )

    def _whnf_step(self, env: Env) -> Term:
        v = env.local_value(self.k)
        if v is None:
            return self
        return v.shift(self.k + 1)


@dataclass(frozen=True)
class MetaVar(Term):
    """Metavariable introduced during elaboration."""

    mid: int
    is_terminal: ClassVar[bool] = True

    def _infer_type(self, env: Env) -> Term:
        raise TypeError("Cannot infer type for metavariable without elaboration state")


@dataclass(frozen=True)
class Lam(Term):
    """Dependent lambda term with an argument type and body."""

    arg_ty: Term
    body: Term = field(metadata={"": TermFieldMeta(binder_count=1)})
    implicit: bool = False

    # Typing -------------------------------------------------------------------
    def _infer_type(self, env: Env) -> Term:
        body_ty = self.body.infer_type(env.push_binder(self.arg_ty))
        return Pi(self.arg_ty, body_ty, implicit=self.implicit)

    def _type_check(self, expected_ty: Term, env: Env) -> None:
        if not isinstance(expected_ty, Pi):
            raise TypeError(
                "Lambda expected to have Pi type:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}"
            )
        if not self.arg_ty._type_equal(expected_ty.arg_ty, env):
            raise TypeError(
                "Lambda domain mismatch:\n"
                f"  term = {self}\n"
                f"  expected domain = {expected_ty.arg_ty}\n"
                f"  found domain = {self.arg_ty}"
            )
        if self.implicit != expected_ty.implicit:
            raise TypeError(
                "Lambda implicitness mismatch:\n"
                f"  term = {self}\n"
                f"  expected implicit = {expected_ty.implicit}\n"
                f"  found implicit = {self.implicit}"
            )
        self.body.type_check(expected_ty.return_ty, env.push_binder(self.arg_ty))


@dataclass(frozen=True)
class Pi(Term):
    """Dependent function type (Pi-type)."""

    arg_ty: Term
    return_ty: Term = field(metadata={"": TermFieldMeta(binder_count=1)})
    implicit: bool = False

    # Typing -------------------------------------------------------------------
    def _infer_type(self, env: Env) -> Term:
        arg_level = self.arg_ty.expect_universe(env)
        body_level = self.return_ty.expect_universe(env.push_binder(self.arg_ty))
        return Univ(arg_level.max(body_level))


@dataclass(frozen=True)
class App(Term):
    """Function application."""

    func: Term
    arg: Term
    implicit: bool = False

    # Reduction ----------------------------------------------------------------
    def _whnf_step(self, env: Env) -> Term:
        f_whnf = self.func.whnf(env)
        if isinstance(f_whnf, Lam) and f_whnf.implicit == self.implicit:
            return f_whnf.body.subst(self.arg)
        return App(f_whnf, self.arg, implicit=self.implicit)

    # Typing -------------------------------------------------------------------
    def _infer_type(self, env: Env) -> Term:
        f_ty = self.func.infer_type(env).whnf(env)
        if not isinstance(f_ty, Pi):
            raise TypeError(
                "Application of non-function:\n"
                f"  term = {self}\n"
                f"  function = {self.func}\n"
                f"  inferred f_ty = {f_ty}"
            )
        if self.implicit != f_ty.implicit:
            raise TypeError(
                "Application implicitness mismatch:\n"
                f"  term = {self}\n"
                f"  expected implicit = {f_ty.implicit}\n"
                f"  found implicit = {self.implicit}"
            )
        self.arg.type_check(f_ty.arg_ty, env)
        return f_ty.return_ty.subst(self.arg)

    def _type_check(self, expected_ty: Term, env: Env) -> None:
        f_ty = self.func.infer_type(env).whnf(env)
        if not isinstance(f_ty, Pi):
            raise TypeError(
                "Application of non-function:\n"
                f"  term = {self}\n"
                f"  function = {self.func}\n"
                f"  inferred f_ty = {f_ty}"
            )
        self.arg.type_check(f_ty.arg_ty, env)
        inferred_ty = f_ty.return_ty.subst(self.arg)
        if not expected_ty._type_equal(inferred_ty, env):
            raise TypeError(
                "Application result type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}\n"
                f"  inferred = {inferred_ty}"
            )


@dataclass(frozen=True, init=False)
class UApp(Term):
    """Universe application for a polymorphic head."""

    head: Term
    levels: tuple[LevelExpr, ...]
    is_terminal: ClassVar[bool] = True

    def __init__(
        self, head: Term, levels: tuple[LevelExpr | int, ...] | LevelExpr | int
    ) -> None:
        if isinstance(levels, tuple):
            level_items = tuple(LevelExpr.of(level) for level in levels)
        else:
            level_items = (LevelExpr.of(levels),)
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "levels", level_items)

    def inst_levels(self, actuals: tuple[LevelExpr, ...]) -> Term:
        return UApp(
            head=self.head,
            levels=tuple(level.instantiate(actuals) for level in self.levels),
        )

    # Reduction ----------------------------------------------------------------
    def _whnf_step(self, env: Env) -> Term:
        from mltt.kernel.env import Const

        if isinstance(self.head, Const):
            decl = env.globals[self.head.name]
            if decl.value is not None and decl.reducible:
                return decl.value.inst_levels(self.levels)
        if isinstance(self.head, Var):
            value = env.local_value(self.head.k)
            if value is not None:
                return value.inst_levels(self.levels)
        return self

    # Typing -------------------------------------------------------------------
    def _infer_type(self, env: Env) -> Term:
        from mltt.kernel.env import Const
        from mltt.kernel.ind import Ind, Ctor

        match self.head:
            case Ind() as ind:
                uarity = ind.uarity
            case Ctor() as ctor:
                uarity = ctor.uarity
            case Const(name):
                uarity = env.globals[name].uarity
            case Var(k):
                uarity = env.binders[k].uarity
            case _:
                raise TypeError(
                    "Universe application head must be a constant, inductive, constructor, or local:\n"
                    f"  head = {self.head}"
                )
        if len(self.levels) != uarity:
            raise TypeError(
                "Universe application arity mismatch:\n"
                f"  head = {self.head}\n"
                f"  expected uarity = {uarity}\n"
                f"  found = {len(self.levels)}"
            )
        head_ty = self.head.infer_type(env)
        return head_ty.inst_levels(self.levels)


@dataclass(frozen=True, init=False)
class Univ(Term):
    """A universe ``Type(level)``."""

    level: LevelExpr = field(default_factory=lambda: LConst(0))
    is_terminal: ClassVar[bool] = True

    def __init__(self, level: LevelExpr | int = 0) -> None:
        level_expr = LevelExpr.of(level)
        object.__setattr__(self, "level", level_expr)
        if isinstance(level_expr, LConst) and level_expr.k < 0:
            raise ValueError("Universe level must be non-negative")

    # Typing -------------------------------------------------------------------
    def _infer_type(self, env: Env) -> Term:
        return Univ(self.level.succ())

    def inst_levels(self, actuals: tuple[LevelExpr, ...]) -> Term:
        return Univ(self.level.instantiate(actuals))

    def _type_check(self, expected_ty: Term, env: Env) -> None:
        if not isinstance(expected_ty, Univ):
            raise TypeError(
                "Universe type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}"
            )
        min_level = self.level.succ()
        if not expected_ty.level >= min_level:
            raise TypeError(
                "Universe level mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}"
            )


@dataclass(frozen=True)
class Let(Term):
    """
    let x : arg_ty := value; body

    `body` is under one binder (x at Var(0)).
    """

    arg_ty: Term
    value: Term
    body: Term = field(metadata={"": TermFieldMeta(binder_count=1)})

    def _infer_type(self, env: Env) -> Term:
        # require annotation for now (no elaboration)
        _ = self.arg_ty.expect_universe(
            env
        )  # or self.arg_ty.infer_type(env).expect_universe(...)
        self.value.type_check(self.arg_ty, env)
        return self.body.infer_type(env.push_let(self.arg_ty, self.value))

    def _whnf_step(self, env: Env) -> Term:
        v1 = self.value.whnf(env)
        if v1 != self.value:
            return Let(arg_ty=self.arg_ty, value=v1, body=self.body)
        return self
