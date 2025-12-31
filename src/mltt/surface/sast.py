"""Surface AST and elaboration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mltt.kernel.ast import App, Lam, Let, Pi, Term, Univ, Var, UApp
from mltt.kernel.environment import Const, Env
from mltt.kernel.levels import LConst, LevelExpr

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
    implicit: bool = False

    def elab(self, env: Env, state: "ElabState") -> tuple[Term, Env]:
        if self.ty is None:
            raise SurfaceError("Missing binder type", self.span)
        ty_term, ty_ty = self.ty.elab_infer(env, state)
        ty_ty_whnf = ty_ty.whnf(env)
        if not isinstance(ty_ty_whnf, Univ):
            raise SurfaceError("Binder type must be a universe", self.span)
        return ty_term, env.push_binder(ty_term, name=self.name)


@dataclass(frozen=True)
class SArg:
    term: SurfaceTerm
    implicit: bool = False


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
            binder = env.binders[idx]
            if binder.uarity:
                levels = tuple(
                    state.fresh_level_meta("implicit", self.span)
                    for _ in range(binder.uarity)
                )
                term = UApp(Var(idx), levels)
                return term, env.local_type(idx).inst_levels(levels)
            return Var(idx), env.local_type(idx)
        decl = env.lookup_global(self.name)
        if decl is not None:
            if decl.uarity:
                levels = tuple(
                    state.fresh_level_meta("implicit", self.span)
                    for _ in range(decl.uarity)
                )
                term = UApp(Const(self.name), levels)
                return term, decl.ty.inst_levels(levels)
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
        if decl.uarity:
            levels = tuple(
                state.fresh_level_meta("implicit", self.span)
                for _ in range(decl.uarity)
            )
            term = UApp(Const(self.name), levels)
            return term, decl.ty.inst_levels(levels)
        return Const(self.name), decl.ty

    def resolve(self, env: Env, names: NameEnv) -> Term:
        if env.lookup_global(self.name) is None:
            raise SurfaceError(f"Unknown constant {self.name}", self.span)
        return Const(self.name)


@dataclass(frozen=True)
class SUniv(SurfaceTerm):
    level: int | None

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        level_expr: LevelExpr | int
        if self.level is None:
            level_expr = state.fresh_level_meta("type", self.span)
        else:
            level_expr = self.level
        term = Univ(level_expr)
        return term, Univ(LevelExpr.of(level_expr).succ())

    def resolve(self, env: Env, names: NameEnv) -> Term:
        if self.level is None:
            raise SurfaceError("Universe level requires elaboration", self.span)
        return Univ(self.level)


@dataclass(frozen=True)
class SAnn(SurfaceTerm):
    term: SurfaceTerm
    ty: SurfaceTerm

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        ty_term, ty_ty = self.ty.elab_infer(env, state)
        ty_ty_whnf = ty_ty.whnf(env)
        if not isinstance(ty_ty_whnf, Univ):
            raise SurfaceError("Annotation must be a universe", self.span)
        term = self.term.elab_check(env, state, ty_term)
        return term, ty_term

    def resolve(self, env: Env, names: NameEnv) -> Term:
        return self.term.resolve(env, names)


@dataclass(frozen=True)
class SHole(SurfaceTerm):
    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        raise SurfaceError("Hole needs expected type", self.span)

    def elab_check(self, env: Env, state: "ElabState", expected: Term) -> Term:
        return state.fresh_meta(env, expected, self.span, kind="hole")

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
        binder_tys, binder_impls, env1 = _elab_binders(env, state, self.binders)
        body_term, body_ty = self.body.elab_infer(env1, state)
        lam_term = body_term
        lam_ty = body_ty
        for ty, implicit in reversed(list(zip(binder_tys, binder_impls))):
            lam_term = Lam(ty, lam_term, implicit=implicit)
            lam_ty = Pi(ty, lam_ty, implicit=implicit)
        return lam_term, lam_ty

    def elab_check(self, env: Env, state: "ElabState", expected: Term) -> Term:
        binder_tys: list[Term] = []
        binder_impls: list[bool] = []
        env1 = env
        expected_ty = expected
        for binder in self.binders:
            pi_ty = expected_ty.whnf(env1)
            if not isinstance(pi_ty, Pi):
                raise SurfaceError("Lambda needs expected function type", self.span)
            if binder.implicit != pi_ty.implicit:
                raise SurfaceError("Lambda binder implicitness mismatch", binder.span)
            if binder.ty is None:
                binder_ty = pi_ty.arg_ty
            else:
                binder_ty, binder_ty_ty = binder.ty.elab_infer(env1, state)
                binder_ty_ty_whnf = binder_ty_ty.whnf(env1)
                if not isinstance(binder_ty_ty_whnf, Univ):
                    raise SurfaceError("Binder type must be a universe", binder.span)
                state.add_constraint(env1, binder_ty, pi_ty.arg_ty, binder.span)
            binder_tys.append(binder_ty)
            binder_impls.append(binder.implicit)
            env1 = env1.push_binder(binder_ty, name=binder.name)
            expected_ty = pi_ty.return_ty
        body_term = self.body.elab_check(env1, state, expected_ty)
        lam_term = body_term
        for ty, implicit in reversed(list(zip(binder_tys, binder_impls))):
            lam_term = Lam(ty, lam_term, implicit=implicit)
        return lam_term

    def resolve(self, env: Env, names: NameEnv) -> Term:
        if any(b.ty is None for b in self.binders):
            raise SurfaceError("Missing binder type", self.span)
        tys: list[Term] = []
        impls: list[bool] = []
        for binder in self.binders:
            assert binder.ty is not None
            tys.append(binder.ty.resolve(env, names))
            impls.append(binder.implicit)
            names.push(binder.name)
        term = self.body.resolve(env, names)
        for ty, implicit in reversed(list(zip(tys, impls))):
            term = Lam(ty, term, implicit=implicit)
            names.pop()
        return term


@dataclass(frozen=True)
class SPi(SurfaceTerm):
    binders: tuple[SBinder, ...]
    body: SurfaceTerm

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        binder_tys, binder_impls, env1 = _elab_binders(env, state, self.binders)
        body_term, body_ty = self.body.elab_infer(env1, state)
        body_ty_whnf = body_ty.whnf(env1)
        if not isinstance(body_ty_whnf, Univ):
            raise SurfaceError("Pi body must be a universe", self.body.span)
        body_level = body_ty_whnf.level
        pi_term: Term = body_term
        result_level: LevelExpr | None = None
        for ty, implicit in reversed(list(zip(binder_tys, binder_impls))):
            arg_ty_ty = ty.infer_type(env).whnf(env)
            if not isinstance(arg_ty_ty, Univ):
                raise SurfaceError("Pi domain must be a universe", self.span)
            arg_level = arg_ty_ty.level
            if result_level is None:
                result_level = state.fresh_level_meta("type", self.span)
            state.add_level_constraint(arg_level, result_level, self.span)
            state.add_level_constraint(body_level, result_level, self.span)
            body_level = result_level
            pi_term = Pi(ty, pi_term, implicit=implicit)
        if result_level is None:
            result_level = state.fresh_level_meta("type", self.span)
            state.add_level_constraint(body_level, result_level, self.span)
        return pi_term, Univ(result_level)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        if any(b.ty is None for b in self.binders):
            raise SurfaceError("Missing binder type", self.span)
        tys: list[Term] = []
        impls: list[bool] = []
        for binder in self.binders:
            assert binder.ty is not None
            tys.append(binder.ty.resolve(env, names))
            impls.append(binder.implicit)
            names.push(binder.name)
        term = self.body.resolve(env, names)
        for ty, implicit in reversed(list(zip(tys, impls))):
            term = Pi(ty, term, implicit=implicit)
            names.pop()
        return term


@dataclass(frozen=True)
class SApp(SurfaceTerm):
    fn: SurfaceTerm
    args: tuple[SArg, ...]

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        fn_term, fn_ty = self.fn.elab_infer(env, state)
        pending = list(self.args)
        while pending:
            fn_ty_whnf = fn_ty.whnf(env)
            if not isinstance(fn_ty_whnf, Pi):
                raise SurfaceError("Application of non-function", pending[0].term.span)
            if fn_ty_whnf.implicit and not pending[0].implicit:
                meta = state.fresh_meta(
                    env, fn_ty_whnf.arg_ty, self.span, kind="implicit"
                )
                fn_term = App(fn_term, meta, implicit=True)
                fn_ty = fn_ty_whnf.return_ty.subst(meta)
                continue
            arg = pending.pop(0)
            if arg.implicit != fn_ty_whnf.implicit:
                raise SurfaceError(
                    "Implicit argument provided where explicit expected", arg.term.span
                )
            arg_term = arg.term.elab_check(env, state, fn_ty_whnf.arg_ty)
            fn_term = App(fn_term, arg_term, implicit=fn_ty_whnf.implicit)
            fn_ty = fn_ty_whnf.return_ty.subst(arg_term)
        while True:
            fn_ty_whnf = fn_ty.whnf(env)
            if not isinstance(fn_ty_whnf, Pi) or not fn_ty_whnf.implicit:
                break
            meta = state.fresh_meta(env, fn_ty_whnf.arg_ty, self.span, kind="implicit")
            fn_term = App(fn_term, meta, implicit=True)
            fn_ty = fn_ty_whnf.return_ty.subst(meta)
        return fn_term, fn_ty

    def resolve(self, env: Env, names: NameEnv) -> Term:
        fn = self.fn.resolve(env, names)
        term = fn
        for arg in self.args:
            arg_term = arg.term.resolve(env, names)
            term = App(term, arg_term, implicit=arg.implicit)
        return term


@dataclass(frozen=True)
class SUApp(SurfaceTerm):
    head: SurfaceTerm
    levels: tuple[int, ...]

    def elab_infer(self, env: Env, state: "ElabState") -> tuple[Term, Term]:
        from mltt.kernel.ast import UApp
        from mltt.surface.sind import SInd, SCtor

        head_term: Term
        match self.head:
            case SVar(name=name):
                if env.lookup_local(name) is not None:
                    raise SurfaceError("UApp head must be a global", self.span)
                decl = env.lookup_global(name)
                if decl is None:
                    raise SurfaceError(f"Unknown identifier {name}", self.span)
                head_term = Const(name)
            case SConst(name=name):
                decl = env.lookup_global(name)
                if decl is None:
                    raise SurfaceError(f"Unknown constant {name}", self.span)
                head_term = Const(name)
            case SInd(name=name):
                decl = env.lookup_global(name)
                if decl is None or decl.value is None:
                    raise SurfaceError(f"Unknown inductive {name}", self.span)
                head_term = decl.value
            case SCtor(name=name):
                decl = env.lookup_global(name)
                if decl is None or decl.value is None:
                    raise SurfaceError(f"Unknown constructor {name}", self.span)
                head_term = decl.value
            case _:
                head_term, _ = self.head.elab_infer(env, state)
                if isinstance(head_term, UApp):
                    head_term = head_term.head
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
        ty_term, ty_ty = self.ty.elab_infer(env, state)
        ty_ty_whnf = ty_ty.whnf(env)
        if not isinstance(ty_ty_whnf, Univ):
            raise SurfaceError("Let type must be a universe", self.span)
        val_term = self.val.elab_check(env, state, ty_term)
        uarity, ty_term, val_term = state.generalize_levels_for_let(ty_term, val_term)
        env1 = env.push_let(ty_term, val_term, name=self.name, uarity=uarity)
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
) -> tuple[list[Term], list[bool], Env]:
    binder_tys: list[Term] = []
    binder_impls: list[bool] = []
    for binder in binders:
        ty_term, env = binder.elab(env, state)
        binder_tys.append(ty_term)
        binder_impls.append(binder.implicit)
    return binder_tys, binder_impls, env
