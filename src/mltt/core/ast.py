"""Object-oriented abstract syntax tree nodes for the miniature MLTT."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace, Field, field
from operator import methodcaller
from typing import TYPE_CHECKING, Any, Callable, ClassVar

if TYPE_CHECKING:
    from .debruijn import Ctx


@dataclass(frozen=True)
class Term:
    """Base class for all MLTT terms."""

    is_terminal: ClassVar[bool] = False

    # --- De Bruijn operations -------------------------------------------------
    @classmethod
    def _reducible_fields(cls) -> tuple[Field, ...]:
        return () if cls.is_terminal else fields(cls)

    @classmethod
    def _checkable_fields(cls) -> tuple[Field, ...]:
        return tuple(
            f
            for f in cls.__dataclass_fields__.values()
            if not f.metadata.get("uncheckable")
        )

    def _map_field(self, field_info: Field, mapper: Callable[[Term, Any], Term]) -> Any:
        value = getattr(self, field_info.name)
        if isinstance(value, Term):
            return mapper(value, field_info.metadata)
        if isinstance(value, tuple):
            mapped_items = []
            for item in value:
                if isinstance(item, Term):
                    mapped_item = mapper(item, field_info.metadata)
                    mapped_items.append(mapped_item)
                else:
                    mapped_items.append(item)
            return tuple(mapped_items)
        return value

    def _replace_terms(self, mapper: Callable[[Term, Any], Term]) -> Term:
        updates = {f.name: self._map_field(f, mapper) for f in self._reducible_fields()}
        # noinspection PyArgumentList
        return replace(self, **updates) if updates else self

    def shift(self, by: int, cutoff: int = 0) -> Term:
        """Shift free variables in the term."""
        return self._replace_terms(lambda t, m: t.shift(by, cutoff + m.get("binds", 0)))

    def subst(self, sub: Term, j: int = 0) -> Term:
        """Substitute ``sub`` for ``Var(j)`` inside the term."""
        return self._replace_terms(
            lambda t, m: t.subst(sub.shift(m.get("binds", 0)), j + m.get("binds", 0))
        )

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
        for f in self._reducible_fields():
            new_value = self._map_field(f, lambda t, _: t._reduce_inside_step(reducer))
            value = getattr(self, f.name)
            if new_value != value:
                # noinspection PyArgumentList
                return replace(self, **{f.name: new_value})
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
    def infer_type(self, ctx: Ctx | None = None) -> Term:
        from .debruijn import Ctx

        return self._infer_type(ctx or Ctx())

    def _infer_type(self, ctx: Ctx) -> Term:
        raise TypeError(f"Unexpected term in infer_type:\n  term = {self!r}")

    def type_check(self, ty: Term, ctx: Ctx | None = None) -> None:
        from .debruijn import Ctx

        self._type_check(ty.whnf(), ctx or Ctx())

    def _type_check(self, expected_ty: Term, ctx: Ctx) -> None:
        self._check_against_inferred(expected_ty, ctx, label=type(self).__name__)

    def expect_universe(self, ctx: Ctx | None = None) -> int:
        ty = self.infer_type(ctx).whnf()
        if not isinstance(ty, Univ):
            raise TypeError(
                "Expected a universe:\n" f"  term = {self}\n" f"  inferred = {ty}"
            )
        return ty.level

    def type_equal(self, other: Term) -> bool:
        self_whnf = self.whnf()
        other_whnf = other.whnf()
        if self_whnf == other_whnf:
            return True
        if type(self_whnf) != type(other_whnf):
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

    def _check_against_inferred(
        self, expected_ty: Term, ctx: Ctx, *, label: str
    ) -> None:
        inferred_ty = self.infer_type(ctx)
        if not expected_ty.type_equal(inferred_ty):
            raise TypeError(
                f"{label} type mismatch:\n"
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
    def _infer_type(self, ctx: Ctx) -> Term:
        if self.k < len(ctx):
            return ctx[self.k].ty.shift(self.k + 1)
        raise TypeError(f"Unbound variable {self.k}")

    def _type_check(self, expected_ty: Term, ctx: Ctx) -> None:
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
    body: Term = field(metadata={"binds": 1})

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx) -> Term:
        body_ty = self.body.infer_type(ctx.push(self.arg_ty))
        return Pi(self.arg_ty, body_ty)

    def _type_check(self, expected_ty: Term, ctx: Ctx) -> None:
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
        self.body.type_check(expected_ty.return_ty, ctx.push(self.arg_ty))


@dataclass(frozen=True)
class Pi(Term):
    """Dependent function type (Pi-type)."""

    arg_ty: Term
    return_ty: Term = field(metadata={"binds": 1})

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx) -> Term:
        arg_level = self.arg_ty.expect_universe(ctx)
        body_level = self.return_ty.expect_universe(ctx.push(self.arg_ty))
        return Univ(max(arg_level, body_level))


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
    def _infer_type(self, ctx: Ctx) -> Term:
        f_ty = self.func.infer_type(ctx).whnf()
        if not isinstance(f_ty, Pi):
            raise TypeError(
                "Application of non-function:\n"
                f"  term = {self}\n"
                f"  function = {self.func}\n"
                f"  inferred f_ty = {f_ty}"
            )
        self.arg.type_check(f_ty.arg_ty, ctx)
        return f_ty.return_ty.subst(self.arg)

    def _type_check(self, expected_ty: Term, ctx: Ctx) -> None:
        f_ty = self.func.infer_type(ctx).whnf()
        if not isinstance(f_ty, Pi):
            raise TypeError(
                "Application of non-function:\n"
                f"  term = {self}\n"
                f"  function = {self.func}\n"
                f"  inferred f_ty = {f_ty}"
            )
        self.arg.type_check(f_ty.arg_ty, ctx)
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

    level: int = 0
    is_terminal: ClassVar[bool] = True

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("Universe level must be non-negative")

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx) -> Term:
        return Univ(self.level + 1)

    def _type_check(self, expected_ty: Term, ctx: Ctx) -> None:
        if not isinstance(expected_ty, Univ):
            raise TypeError(
                "Universe type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}"
            )
        # TODO: Check Universe Levels once Ind supports cumulativity


__all__ = [
    "Term",
    "Var",
    "Lam",
    "Pi",
    "App",
    "Univ",
]
