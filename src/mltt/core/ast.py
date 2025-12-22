"""Object-oriented abstract syntax tree nodes for the miniature MLTT."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace, Field
from operator import methodcaller
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Self

if TYPE_CHECKING:
    from .debruijn import Ctx


# --- Structural utilities -----------------------------------------------------
def _map_value(value: Any, mapper: Reducer) -> tuple[Any, bool]:
    if isinstance(value, Term):
        mapped = mapper(value)
        return mapped, mapped != value
    if isinstance(value, tuple):
        changed = False
        mapped_items = []
        for item in value:
            if isinstance(item, Term):
                mapped_item = mapper(item)
                changed = changed or mapped_item != item
                mapped_items.append(mapped_item)
            else:
                mapped_items.append(item)
        return (tuple(mapped_items), True) if changed else (value, False)
    return value, False


def _reduce_value(value: Any, reducer: Reducer) -> tuple[Any, bool]:
    if isinstance(value, Term):
        reduced = value.reduce_inside_step(reducer)
        return reduced, reduced != value
    if isinstance(value, tuple):
        changed = False
        items: list[Any] = []
        for item in value:
            if isinstance(item, Term):
                reduced = item.reduce_inside_step(reducer)
                changed = changed or reduced != item
                items.append(reduced)
            else:
                items.append(item)
        return (tuple(items), True) if changed else (value, False)
    return value, False


@dataclass(frozen=True)
class Term:
    """Base class for all MLTT terms."""

    is_terminal: ClassVar[bool] = False

    # --- De Bruijn operations -------------------------------------------------
    def _reducible_fields(self) -> tuple[Field, ...]:
        return () if self.is_terminal else fields(self)

    def _replace_terms(self, mapper: Reducer) -> Term:
        updates: dict[str, Any] = {}
        for field_info in self._reducible_fields():
            value = getattr(self, field_info.name)
            mapped, changed = _map_value(value, mapper)
            if changed:
                updates[field_info.name] = mapped
        # noinspection PyArgumentList
        return replace(self, **updates) if updates else self

    def shift(self, by: int, cutoff: int = 0) -> Term:
        """Shift free variables in the term."""
        return self._replace_terms(lambda child: child.shift(by, cutoff))

    def subst(self, sub: Term, j: int = 0) -> Term:
        """Substitute ``sub`` for ``Var(j)`` inside the term."""
        return self._replace_terms(lambda child: child.subst(sub, j))

    # --- Reduction ------------------------------------------------------------
    def whnf(self) -> Term:
        """Weak head normal form."""
        return self

    def reduce_inside_step(self, reducer: Reducer) -> Term:
        reduced = reducer(self)
        if reduced != self:
            return reduced
        return self._reduce_children(reducer)

    def _reduce_children(self, reducer: Reducer) -> Term:
        for field_info in self._reducible_fields():
            value = getattr(self, field_info.name)
            new_value, changed = _reduce_value(value, reducer)
            if changed:
                # noinspection PyArgumentList
                return replace(self, **{field_info.name: new_value})
        return self

    def normalize_step(self) -> Term:
        """Perform a single beta/iota reduction step anywhere in the term."""
        return self.reduce_inside_step(methodcaller("whnf"))

    def normalize(self) -> Term:
        """Fully normalize the term."""
        term: Term = self
        while True:
            next_term = term.normalize_step()
            if next_term == term:
                return term
            term = next_term

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

    def type_equal(self, other: Term, ctx: Ctx | None = None) -> bool:
        from .debruijn import Ctx

        ctx = ctx or Ctx()
        self_whnf = self.whnf()
        other_whnf = other.whnf()
        if self_whnf == other_whnf:
            return True
        if type(self_whnf) != type(other_whnf):
            return False
        return self_whnf._type_equal_with(other_whnf, ctx)

    def _type_equal_with(self, other: Self, ctx: Ctx) -> bool:
        return False

    def _check_against_inferred(
        self, expected_ty: Term, ctx: Ctx, *, label: str
    ) -> None:
        inferred_ty = self.infer_type(ctx)
        if not expected_ty.type_equal(inferred_ty, ctx):
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
        if not found_ty.type_equal(expected_ty, ctx):
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
    body: Term

    # De Bruijn ---------------------------------------------------------------
    def shift(self, by: int, cutoff: int = 0) -> Term:
        return Lam(self.arg_ty.shift(by, cutoff), self.body.shift(by, cutoff + 1))

    def subst(self, sub: Term, j: int = 0) -> Term:
        return Lam(
            self.arg_ty.subst(sub, j),
            self.body.subst(sub.shift(1), j + 1),
        )

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx) -> Term:
        body_ty = self.body.infer_type(ctx.insert(self.arg_ty))
        return Pi(self.arg_ty, body_ty)

    def _type_check(self, expected_ty: Term, ctx: Ctx) -> None:
        if not isinstance(expected_ty, Pi):
            raise TypeError(
                "Lambda expected to have Pi type:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}"
            )
        if not self.arg_ty.type_equal(expected_ty.arg_ty, ctx):
            raise TypeError(
                "Lambda domain mismatch:\n"
                f"  term = {self}\n"
                f"  expected domain = {expected_ty.arg_ty}\n"
                f"  found domain = {self.arg_ty}"
            )
        ctx1 = ctx.insert(self.arg_ty)
        self.body.type_check(expected_ty.return_ty, ctx1)

    def _type_equal_with(self, other: Self, ctx: Ctx) -> bool:
        return self.arg_ty.type_equal(other.arg_ty, ctx) and self.body.type_equal(
            other.body, ctx.insert(self.arg_ty)
        )


@dataclass(frozen=True)
class Pi(Term):
    """Dependent function type (Pi-type)."""

    arg_ty: Term
    return_ty: Term

    # De Bruijn ---------------------------------------------------------------
    def shift(self, by: int, cutoff: int = 0) -> Term:
        return Pi(self.arg_ty.shift(by, cutoff), self.return_ty.shift(by, cutoff + 1))

    def subst(self, sub: Term, j: int = 0) -> Term:
        return Pi(
            self.arg_ty.subst(sub, j),
            self.return_ty.subst(sub.shift(1), j + 1),
        )

    # Typing -------------------------------------------------------------------
    def _infer_type(self, ctx: Ctx) -> Term:
        arg_level = self.arg_ty.expect_universe(ctx)
        body_level = self.return_ty.expect_universe(ctx.insert(self.arg_ty))
        return Univ(max(arg_level, body_level))

    def _type_equal_with(self, other: Self, ctx: Ctx) -> bool:
        return self.arg_ty.type_equal(other.arg_ty, ctx) and self.return_ty.type_equal(
            other.return_ty, ctx.insert(self.arg_ty)
        )


@dataclass(frozen=True)
class App(Term):
    """Function application."""

    func: Term
    arg: Term

    # Reduction ----------------------------------------------------------------
    def whnf(self) -> Term:
        f_whnf = self.func.whnf()
        if isinstance(f_whnf, Lam):
            return f_whnf.body.subst(self.arg).whnf()
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
        if not expected_ty.type_equal(inferred_ty, ctx):
            raise TypeError(
                "Application result type mismatch:\n"
                f"  term = {self}\n"
                f"  expected = {expected_ty}\n"
                f"  inferred = {inferred_ty}"
            )

    def _type_equal_with(self, other: Self, ctx: Ctx) -> bool:
        return self.func.type_equal(other.func, ctx) and self.arg.type_equal(
            other.arg, ctx
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


Reducer = Callable[[Term], Term]

__all__ = [
    "Term",
    "Var",
    "Lam",
    "Pi",
    "App",
    "Univ",
    "Reducer",
]
