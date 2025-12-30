"""Name resolution for surface terms with explicit binder types."""

from __future__ import annotations

from dataclasses import dataclass

from mltt.kernel.ast import App, Lam, Let, Pi, Term, Univ, Var
from mltt.kernel.environment import Const, Env
from mltt.kernel.telescope import mk_app
from mltt.surface.syntax import (
    SurfaceTerm,
    SurfaceError,
    SBinder,
    SVar,
    SType,
    SAnn,
    SLam,
    SPi,
    SApp,
    SLet,
)


@dataclass
class NameEnv:
    locals: list[str]

    @staticmethod
    def empty() -> NameEnv:
        return NameEnv([])

    def push(self, name: str) -> None:
        self.locals.insert(0, name)

    def pop(self) -> None:
        self.locals.pop(0)

    def lookup(self, name: str) -> int | None:
        try:
            return self.locals.index(name)
        except ValueError:
            return None


def resolve_term(env: Env, names: NameEnv, term: SurfaceTerm) -> Term:
    if isinstance(term, SVar):
        idx = names.lookup(term.name)
        if idx is not None:
            return Var(idx)
        if env.lookup_global(term.name) is not None:
            return Const(term.name)
        raise SurfaceError(f"Unknown identifier {term.name}", term.span)
    if isinstance(term, SType):
        return Univ(term.level)
    if isinstance(term, SAnn):
        return resolve_term(env, names, term.term)
    if isinstance(term, SLam):
        return _resolve_lam(env, names, term.binders, term.body)
    if isinstance(term, SPi):
        return _resolve_pi(env, names, term.binders, term.body)
    if isinstance(term, SApp):
        fn = resolve_term(env, names, term.fn)
        args = tuple(resolve_term(env, names, arg) for arg in term.args)
        return mk_app(fn, *args)
    if isinstance(term, SLet):
        ty = resolve_term(env, names, term.ty)
        val = resolve_term(env, names, term.val)
        names.push(term.name)
        body = resolve_term(env, names, term.body)
        names.pop()
        return Let(ty, val, body)
    raise SurfaceError("Unsupported surface term", term.span)


def _resolve_binder(env: Env, names: NameEnv, binder: SBinder) -> Term:
    if binder.ty is None:
        raise SurfaceError("Missing binder type", binder.span)
    return resolve_term(env, names, binder.ty)


def _resolve_pi(
    env: Env, names: NameEnv, binders: tuple[SBinder, ...], body: SurfaceTerm
) -> Term:
    tys: list[Term] = []
    for binder in binders:
        tys.append(_resolve_binder(env, names, binder))
        names.push(binder.name)
    term: Term = resolve_term(env, names, body)
    for ty in reversed(tys):
        term = Pi(ty, term)
        names.pop()
    return term


def _resolve_lam(
    env: Env, names: NameEnv, binders: tuple[SBinder, ...], body: SurfaceTerm
) -> Term:
    tys: list[Term] = []
    for binder in binders:
        tys.append(_resolve_binder(env, names, binder))
        names.push(binder.name)
    term: Term = resolve_term(env, names, body)
    for ty in reversed(tys):
        term = Lam(ty, term)
        names.pop()
    return term
