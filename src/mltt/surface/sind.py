"""Surface inductive references."""

from __future__ import annotations

from dataclasses import dataclass

from mltt.kernel.ast import Term
from mltt.kernel.environment import Env
from mltt.kernel.ind import Ctor, Ind
from mltt.surface.elab_state import ElabState
from mltt.surface.sast import NameEnv, SurfaceError, SurfaceTerm


@dataclass(frozen=True)
class SInd(SurfaceTerm):
    name: str

    def elab_infer(self, env: Env, state: ElabState) -> tuple[Term, Term]:
        decl = env.lookup_global(self.name)
        if decl is None or decl.value is None:
            raise SurfaceError(f"Unknown inductive {self.name}", self.span)
        if not isinstance(decl.value, Ind):
            raise SurfaceError(f"{self.name} is not an inductive", self.span)
        return decl.value, decl.value.infer_type(env)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Inductive references require elaboration", self.span)


@dataclass(frozen=True)
class SCtor(SurfaceTerm):
    name: str

    def elab_infer(self, env: Env, state: ElabState) -> tuple[Term, Term]:
        decl = env.lookup_global(self.name)
        if decl is None or decl.value is None:
            raise SurfaceError(f"Unknown constructor {self.name}", self.span)
        if not isinstance(decl.value, Ctor):
            raise SurfaceError(f"{self.name} is not a constructor", self.span)
        return decl.value, decl.value.infer_type(env)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Constructor references require elaboration", self.span)
