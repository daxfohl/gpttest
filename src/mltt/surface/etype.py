"""Elaboration-only type wrappers with implicit binder metadata."""

from __future__ import annotations

from dataclasses import dataclass, field

from mltt.kernel.ast import Term
from mltt.kernel.env import Env, GlobalDecl


@dataclass(frozen=True)
class ElabType:
    """Kernel type plus implicit binder flags for surface elaboration."""

    term: Term
    implicit_spine: tuple[bool, ...] = ()


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

    def local_type(self, k: int) -> ElabType:
        if k < 0 or k >= len(self.locals):
            raise IndexError(f"Unbound variable {k}")
        local = self.locals[k]
        return ElabType(local.term.shift(k + 1), local.implicit_spine)

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
