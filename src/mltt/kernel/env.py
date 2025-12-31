from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import overload

from mltt.kernel.ast import Term


@dataclass(frozen=True)
class GlobalDecl:
    """
    A top-level declaration.

    - ty: type of the constant/definition
    - value: definitional body (None for axioms/constructors if you don't unfold them)
    - uarity: number of universe binders in the declaration
    """

    ty: Term
    value: Term | None = None
    reducible: bool = True
    uarity: int = 0


@dataclass(frozen=True)
class Binder:
    """Single context entry containing the type of a bound variable."""

    ty: Term
    name: str | None = None
    value: Term | None = None  # for let-bindings; None for ordinary binders
    uarity: int = 0

    @overload
    @staticmethod
    def of(entry: Binder) -> Binder: ...

    @overload
    @staticmethod
    def of(
        entry: Term,
        name: str | None = None,
        value: Term | None = None,
        uarity: int = 0,
    ) -> Binder: ...

    @staticmethod
    def of(
        entry: Binder | Term,
        name: str | None = None,
        value: Term | None = None,
        uarity: int = 0,
    ) -> Binder:
        """Coerce an entry or term into a ``Binder``."""

        if isinstance(entry, Binder):
            assert name is None and value is None and uarity == 0
            return entry
        return Binder(entry, name, value, uarity)


@dataclass(frozen=True)
class Env:
    """
    Surface/elaboration environment over a de Bruijn typing context.

    Notes:
        - Binder types scoped in tail context.
    """

    binders: tuple[Binder, ...] = ()
    globals: MappingProxyType[str, GlobalDecl] = MappingProxyType({})

    # ---- extending the environment ----
    def push_binder(
        self,
        ty: Term,
        name: str | None = None,
        uarity: int = 0,
    ) -> Env:
        """
        Push a new binder at de Bruijn index 0.

        This mirrors Ctx.push(ty) which does not rewrite existing entry types.
        """
        binders = (Binder.of(ty, name, uarity=uarity),) + self.binders
        return replace(self, binders=binders)

    def push_binders(self, *binders: Term | tuple[Term, str]) -> Env:
        """
        Push many binders ordered outermost -> innermost (like nested_lam/nested_pi).

        Example:
            env.push_binders(("x", A), ("y", B)) pushes x then y, so y ends up at index 0.
        """
        env = self
        for b in binders:
            if isinstance(b, (tuple, list)):
                env = env.push_binder(*b)
            else:
                env = env.push_binder(b)
        return env

    def push_let(
        self,
        ty: Term,
        value: Term,
        name: str | None = None,
        uarity: int = 0,
    ) -> Env:
        """
        Push a let-bound variable at index 0 with its type and definitional value.

        Whether the kernel has let-terms or you treat lets as sugar,
        this is still useful for pretty-printing and optional unfolding.
        """

        binders = (Binder.of(ty, name, value, uarity),) + self.binders
        return replace(self, binders=binders)

    # ---- name resolution (locals) ----
    def lookup_local(self, name: str) -> int | None:
        """
        Return the de Bruijn index for the nearest local binder with this name.
        """
        # index 0 = innermost
        for i, le in enumerate(self.binders):
            if le.name == name:
                return i
        return None

    # ---- lookup (globals) ----
    def lookup_global(self, name: str) -> GlobalDecl | None:
        return self.globals.get(name)

    def lookup_name(self, name: str) -> tuple[str, int | GlobalDecl] | None:
        """
        Unified lookup for surface resolution:
          - if local found: ("local", index)
          - else if global found: ("global", decl)
          - else None
        """
        k = self.lookup_local(name)
        if k is not None:
            return "local", k
        g = self.lookup_global(name)
        if g is not None:
            return "global", g
        return None

    # ---- helpers for typing + printing ----
    def local_type(self, k: int) -> Term:
        """
        Return the type of Var(k) in this environment in *current scope*.

        Stored entry types are scoped in tail, and we shift on lookup by k.
        """
        if k < 0 or k >= len(self.binders):
            raise IndexError(f"Unbound variable {k}")
        return self.binders[k].ty.shift(k + 1)

    def local_value(self, k: int) -> Term | None:
        if k < 0:
            raise IndexError(f"Unbound variable {k}")
        if k >= len(self.binders):
            return None
        return self.binders[k].value

    def names(self) -> tuple[str | None, ...]:
        """
        Names ordered by de Bruijn index (0 = innermost).
        """
        return tuple(le.name for le in self.binders)

    @staticmethod
    def of(*env: Binder | Term) -> Env:
        return Env(tuple(Binder.of(entry) for entry in env))

    def __str__(self) -> str:
        if len(self.binders) < 2:
            return f"Ctx{self.binders}"
        return f"Ctx(\n{"".join([f"  #{i}: {e.ty}\n" for i, e in enumerate(self.binders)])})"


@dataclass(frozen=True)
class Const(Term):
    name: str

    def _infer_type(self, env: Env) -> Term:
        decl = env.globals[self.name]
        return decl.ty

    def _whnf_step(self, env: Env) -> Term:
        decl = env.globals[self.name]
        if decl.value is None or not decl.reducible:
            return self
        return decl.value
