"""Surface AST and elaboration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mltt.kernel.ast import App, Lam, Let, Pi, Term, Univ, Var
from mltt.kernel.environment import Const, Env
from mltt.kernel.levels import LConst
from mltt.kernel.telescope import mk_app, mk_pis

if TYPE_CHECKING:
    from mltt.surface.elab_state import ElabState


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def extract(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass
class SurfaceError(Exception):
    message: str
    span: Span
    source: str | None = None

    def __str__(self) -> str:
        if self.source is None:
            return f"{self.message} @ {self.span.start}:{self.span.end}"
        snippet = self.span.extract(self.source)
        return f"{self.message} @ {self.span.start}:{self.span.end}: {snippet!r}"


@dataclass(frozen=True)
class SurfaceTerm:
    span: Span

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        raise SurfaceError("Unsupported surface term", self.span)

    def elab_check(self, env: Env, state: "ElabState", expected: Term) -> Term:
        term, term_ty = self.elab_infer(env, state)
        state.add_constraint(env, term_ty, expected, self.span)
        return term

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Unsupported surface term", self.span)


@dataclass(frozen=True)
class SBinder:
    name: str
    ty: SurfaceTerm | None
    span: Span

    def elab(self, env: Env, state: "ElabState") -> tuple[Term, Env]:
        if self.ty is None:
            raise SurfaceError("Missing binder type", self.span)
        ty_term, _ = self.ty.elab_infer(env, state)
        _ = ty_term.expect_universe(env)
        return ty_term, env.push_binder(ty_term, name=self.name)


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


@dataclass(frozen=True)
class SVar(SurfaceTerm):
    name: str

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        idx = env.lookup_local(self.name)
        if idx is not None:
            return Var(idx), env.local_type(idx)
        decl = env.lookup_global(self.name)
        if decl is not None:
            return Const(self.name), decl.ty
        raise SurfaceError(f"Unknown identifier {self.name}", self.span)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        idx = names.lookup(self.name)
        if idx is not None:
            return Var(idx)
        if env.lookup_global(self.name) is not None:
            return Const(self.name)
        raise SurfaceError(f"Unknown identifier {self.name}", self.span)


@dataclass(frozen=True)
class SConst(SurfaceTerm):
    name: str

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        decl = env.lookup_global(self.name)
        if decl is None:
            raise SurfaceError(f"Unknown constant {self.name}", self.span)
        return Const(self.name), decl.ty

    def resolve(self, env: Env, names: NameEnv) -> Term:
        if env.lookup_global(self.name) is None:
            raise SurfaceError(f"Unknown constant {self.name}", self.span)
        return Const(self.name)


@dataclass(frozen=True)
class SUniv(SurfaceTerm):
    level: int

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        term = Univ(self.level)
        return term, Univ(self.level + 1)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        return Univ(self.level)


@dataclass(frozen=True)
class SAnn(SurfaceTerm):
    term: SurfaceTerm
    ty: SurfaceTerm

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        ty_term, _ = self.ty.elab_infer(env, state)
        _ = ty_term.expect_universe(env)
        term = self.term.elab_check(env, state, ty_term)
        return term, ty_term

    def resolve(self, env: Env, names: NameEnv) -> Term:
        return self.term.resolve(env, names)


@dataclass(frozen=True)
class SHole(SurfaceTerm):
    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        raise SurfaceError("Hole needs expected type", self.span)

    def elab_check(self, env: Env, state: "ElabState", expected: Term) -> Term:
        return state.fresh_meta(env, expected, self.span)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Hole requires elaboration", self.span)


@dataclass(frozen=True)
class SLam(SurfaceTerm):
    binders: tuple[SBinder, ...]
    body: SurfaceTerm

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        if any(b.ty is None for b in self.binders):
            raise SurfaceError(
                "Cannot infer unannotated lambda; add binder types or use check-mode",
                self.span,
            )
        binder_tys, env1 = _elab_binders(env, state, self.binders)
        body_term, body_ty = self.body.elab_infer(env1, state)
        lam_term = body_term
        lam_ty = body_ty
        for ty in reversed(binder_tys):
            lam_term = Lam(ty, lam_term)
            lam_ty = Pi(ty, lam_ty)
        return lam_term, lam_ty

    def elab_check(self, env: Env, state: "ElabState", expected: Term) -> Term:
        binder_tys: list[Term] = []
        env1 = env
        expected_ty = expected
        for binder in self.binders:
            pi_ty = expected_ty.whnf(env1)
            if not isinstance(pi_ty, Pi):
                raise SurfaceError("Lambda needs expected function type", self.span)
            if binder.ty is None:
                binder_ty = pi_ty.arg_ty
            else:
                binder_ty, _ = binder.ty.elab_infer(env1, state)
                _ = binder_ty.expect_universe(env1)
                if not binder_ty.type_equal(pi_ty.arg_ty, env1):
                    raise SurfaceError("Lambda binder type mismatch", binder.span)
            binder_tys.append(binder_ty)
            env1 = env1.push_binder(binder_ty, name=binder.name)
            expected_ty = pi_ty.return_ty
        body_term = self.body.elab_check(env1, state, expected_ty)
        lam_term = body_term
        for ty in reversed(binder_tys):
            lam_term = Lam(ty, lam_term)
        return lam_term

    def resolve(self, env: Env, names: NameEnv) -> Term:
        if any(b.ty is None for b in self.binders):
            raise SurfaceError("Missing binder type", self.span)
        tys: list[Term] = []
        for binder in self.binders:
            assert binder.ty is not None
            tys.append(binder.ty.resolve(env, names))
            names.push(binder.name)
        term = self.body.resolve(env, names)
        for ty in reversed(tys):
            term = Lam(ty, term)
            names.pop()
        return term


@dataclass(frozen=True)
class SPi(SurfaceTerm):
    binders: tuple[SBinder, ...]
    body: SurfaceTerm

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        binder_tys, env1 = _elab_binders(env, state, self.binders)
        body_term, _ = self.body.elab_infer(env1, state)
        pi_term = mk_pis(*binder_tys, return_ty=body_term)
        return pi_term, pi_term.infer_type(env)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        if any(b.ty is None for b in self.binders):
            raise SurfaceError("Missing binder type", self.span)
        tys: list[Term] = []
        for binder in self.binders:
            assert binder.ty is not None
            tys.append(binder.ty.resolve(env, names))
            names.push(binder.name)
        term = self.body.resolve(env, names)
        for ty in reversed(tys):
            term = Pi(ty, term)
            names.pop()
        return term


@dataclass(frozen=True)
class SApp(SurfaceTerm):
    fn: SurfaceTerm
    args: tuple[SurfaceTerm, ...]

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        fn_term, fn_ty = self.fn.elab_infer(env, state)
        for arg in self.args:
            fn_ty_whnf = fn_ty.whnf(env)
            if not isinstance(fn_ty_whnf, Pi):
                raise SurfaceError("Application of non-function", arg.span)
            arg_term = arg.elab_check(env, state, fn_ty_whnf.arg_ty)
            fn_term = App(fn_term, arg_term)
            fn_ty = fn_ty_whnf.return_ty.subst(arg_term)
        return fn_term, fn_ty

    def resolve(self, env: Env, names: NameEnv) -> Term:
        fn = self.fn.resolve(env, names)
        args = tuple(arg.resolve(env, names) for arg in self.args)
        return mk_app(fn, *args)


@dataclass(frozen=True)
class SUApp(SurfaceTerm):
    head: SurfaceTerm
    levels: tuple[int, ...]

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        from mltt.kernel.ast import UApp

        head_term, _ = self.head.elab_infer(env, state)
        level_terms = tuple(LConst(level) for level in self.levels)
        uapp = UApp(head_term, level_terms)
        return uapp, uapp.infer_type(env)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        from mltt.kernel.ast import UApp

        head = self.head.resolve(env, names)
        levels = tuple(LConst(level) for level in self.levels)
        return UApp(head, levels)


@dataclass(frozen=True)
class SLet(SurfaceTerm):
    name: str
    ty: SurfaceTerm
    val: SurfaceTerm
    body: SurfaceTerm

    def elab_infer(self, env: Env, state: ElabState) -> tuple[Term, Term]:
        ty_term, _ = self.ty.elab_infer(env, state)
        _ = ty_term.expect_universe(env)
        val_term = self.val.elab_check(env, state, ty_term)
        env1 = env.push_let(ty_term, val_term, name=self.name)
        body_term, body_ty = self.body.elab_infer(env1, state)
        return Let(ty_term, val_term, body_term), body_ty

    def resolve(self, env: Env, names: NameEnv) -> Term:
        ty = self.ty.resolve(env, names)
        val = self.val.resolve(env, names)
        names.push(self.name)
        body = self.body.resolve(env, names)
        names.pop()
        return Let(ty, val, body)


def _elab_binders(
    env: Env, state: "ElabState", binders: tuple[SBinder, ...]
) -> tuple[list[Term], Env]:
    binder_tys: list[Term] = []
    for binder in binders:
        ty_term, env = binder.elab(env, state)
        binder_tys.append(ty_term)
    return binder_tys, env
