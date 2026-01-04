"""Elaboration-only type wrappers with binder metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from mltt.kernel.ast import Term
from mltt.kernel.env import Env, GlobalDecl


@dataclass(frozen=True)
class BinderSpec:
    """Binder metadata (names + implicitness + optional type) for elaboration."""

    name: str | None = None
    implicit: bool = False
    ty: Term | None = None

    def shift(self, amount: int) -> "BinderSpec":
        if amount == 0 or self.ty is None:
            return self
        return BinderSpec(self.name, self.implicit, self.ty.shift(amount))

    def inst_levels(self, levels: tuple) -> "BinderSpec":
        if not levels or self.ty is None:
            return self
        return BinderSpec(self.name, self.implicit, self.ty.inst_levels(levels))


def normalize_binder_name(name: str | None) -> str | None:
    if name == "_":
        return None
    return name


@dataclass(frozen=True)
class ElabType:
    """Kernel type plus binder metadata for surface elaboration."""

    term: Term
    binders: tuple[BinderSpec, ...] = ()

    def whnf(self, env: Env) -> Term:
        return self.term.whnf(env)

    def inst_levels(self, levels: tuple) -> "ElabType":
        if not levels:
            return self
        return ElabType(
            self.term.inst_levels(levels),
            tuple(b.inst_levels(levels) for b in self.binders),
        )

    def shift(self, amount: int) -> "ElabType":
        if amount == 0:
            return self
        return ElabType(
            self.term.shift(amount),
            tuple(b.shift(amount) for b in self.binders),
        )


@dataclass(frozen=True)
class ElabEnv:
    """Surface elaboration environment with implicit binder metadata."""

    kenv: Env
    locals: tuple[ElabType, ...] = ()
    eglobals: dict[str, ElabType] = field(default_factory=dict)

    @staticmethod
    def from_env(env: Env) -> ElabEnv:
        globals_types = {name: ElabType(decl.ty) for name, decl in env.globals.items()}
        return ElabEnv(kenv=env, locals=(), eglobals=globals_types)

    @property
    def binders(self) -> tuple:
        return self.kenv.binders

    def lookup_local(self, name: str) -> int | None:
        return self.kenv.lookup_local(name)

    def lookup_global(self, name: str) -> GlobalDecl | None:
        return self.kenv.lookup_global(name)

    def global_type(self, name: str) -> ElabType | None:
        return self.eglobals.get(name)

    def global_info(self, name: str) -> tuple[GlobalDecl, ElabType] | None:
        decl = self.lookup_global(name)
        if decl is None:
            return None
        gty = self.global_type(name)
        assert gty is not None
        return decl, gty

    def local_type(self, k: int) -> ElabType:
        if k < 0 or k >= len(self.locals):
            raise IndexError(f"Unbound variable {k}")
        local = self.locals[k]
        return local.shift(k + 1)

    def push_binder(
        self, ty: ElabType, name: str | None = None, uarity: int = 0
    ) -> ElabEnv:
        return ElabEnv(
            kenv=self.kenv.push_binder(ty.term, name=name, uarity=uarity),
            locals=(ty,) + self.locals,
            eglobals=self.eglobals,
        )

    def push_let(
        self,
        ty: ElabType,
        value: Term,
        name: str | None = None,
        uarity: int = 0,
    ) -> ElabEnv:
        return ElabEnv(
            kenv=self.kenv.push_let(ty.term, value, name=name, uarity=uarity),
            locals=(ty,) + self.locals,
            eglobals=self.eglobals,
        )
