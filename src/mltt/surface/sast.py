"""Surface AST and elaboration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mltt.kernel.ast import App, Lam, Let, Pi, Term, Univ, Var, UApp
from mltt.kernel.env import Const, Env
from mltt.kernel.levels import LConst, LevelExpr
from mltt.surface.etype import ElabEnv, ElabType

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

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        raise SurfaceError("Unsupported surface term", self.span)

    def elab_check(self, env: ElabEnv, state: "ElabState", expected: ElabType) -> Term:
        term, term_ty = self.elab_infer(env, state)
        state.add_constraint(env.kenv, term_ty.term, expected.term, self.span)
        return term

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Unsupported surface term", self.span)


@dataclass(frozen=True)
class SBinder:
    name: str
    ty: SurfaceTerm | None
    span: Span
    implicit: bool = False

    def elab(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabEnv]:
        if self.ty is None:
            raise SurfaceError("Missing binder type", self.span)
        ty_term, ty_ty = self.ty.elab_infer(env, state)
        ty_ty_whnf = ty_ty.term.whnf(env.kenv)
        if not isinstance(ty_ty_whnf, Univ):
            raise SurfaceError("Binder type must be a universe", self.span)
        implicit_spine = _implicit_spine(self.ty)
        return ty_term, env.push_binder(
            ElabType(ty_term, implicit_spine), name=self.name
        )


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

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        idx = env.lookup_local(self.name)
        if idx is not None:
            binder = env.binders[idx]
            if binder.uarity:
                levels = tuple(
                    state.fresh_level_meta("implicit", self.span)
                    for _ in range(binder.uarity)
                )
                term = UApp(Var(idx), levels)
                return term, ElabType(
                    env.local_type(idx).term.inst_levels(levels),
                    env.local_type(idx).implicit_spine,
                )
            return Var(idx), env.local_type(idx)
        decl = env.lookup_global(self.name)
        if decl is not None:
            if decl.uarity:
                levels = tuple(
                    state.fresh_level_meta("implicit", self.span)
                    for _ in range(decl.uarity)
                )
                term = UApp(Const(self.name), levels)
                gty = env.global_type(self.name)
                assert gty is not None
                return term, ElabType(gty.term.inst_levels(levels), gty.implicit_spine)
            gty = env.global_type(self.name)
            assert gty is not None
            return Const(self.name), gty
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

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        decl = env.lookup_global(self.name)
        if decl is None:
            raise SurfaceError(f"Unknown constant {self.name}", self.span)
        if decl.uarity:
            levels = tuple(
                state.fresh_level_meta("implicit", self.span)
                for _ in range(decl.uarity)
            )
            term = UApp(Const(self.name), levels)
            gty = env.global_type(self.name)
            assert gty is not None
            return term, ElabType(gty.term.inst_levels(levels), gty.implicit_spine)
        gty = env.global_type(self.name)
        assert gty is not None
        return Const(self.name), gty

    def resolve(self, env: Env, names: NameEnv) -> Term:
        if env.lookup_global(self.name) is None:
            raise SurfaceError(f"Unknown constant {self.name}", self.span)
        return Const(self.name)


@dataclass(frozen=True)
class SUniv(SurfaceTerm):
    level: int | str | None

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        level_expr: LevelExpr | int
        if self.level is None:
            level_expr = state.fresh_level_meta("type", self.span)
        elif isinstance(self.level, str):
            level_expr = state.lookup_level(self.level, self.span)
        else:
            level_expr = self.level
        term = Univ(level_expr)
        return term, ElabType(Univ(LevelExpr.of(level_expr).succ()))

    def resolve(self, env: Env, names: NameEnv) -> Term:
        if self.level is None:
            raise SurfaceError("Universe level requires elaboration", self.span)
        if isinstance(self.level, str):
            raise SurfaceError("Universe level requires elaboration", self.span)
        return Univ(self.level)


@dataclass(frozen=True)
class SAnn(SurfaceTerm):
    term: SurfaceTerm
    ty: SurfaceTerm

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        ty_term, ty_ty = self.ty.elab_infer(env, state)
        ty_ty_whnf = ty_ty.term.whnf(env.kenv)
        if not isinstance(ty_ty_whnf, Univ):
            raise SurfaceError("Annotation must be a universe", self.span)
        term = self.term.elab_check(env, state, ElabType(ty_term))
        return term, ElabType(ty_term)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        return self.term.resolve(env, names)


@dataclass(frozen=True)
class SHole(SurfaceTerm):
    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        raise SurfaceError("Hole needs expected type", self.span)

    def elab_check(self, env: ElabEnv, state: "ElabState", expected: ElabType) -> Term:
        return state.fresh_meta(env.kenv, expected.term, self.span, kind="hole")

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Hole requires elaboration", self.span)


@dataclass(frozen=True)
class SLam(SurfaceTerm):
    binders: tuple[SBinder, ...]
    body: SurfaceTerm

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        if any(b.ty is None for b in self.binders):
            raise SurfaceError(
                "Cannot infer unannotated lambda; add binder types or use check-mode",
                self.span,
            )
        binder_tys, binder_impls, env1 = _elab_binders(env, state, self.binders)
        body_term, body_ty = self.body.elab_infer(env1, state)
        lam_term = body_term
        lam_ty_term = body_ty.term
        implicit_spine = body_ty.implicit_spine
        for ty, implicit in reversed(list(zip(binder_tys, binder_impls))):
            lam_term = Lam(ty, lam_term)
            lam_ty_term = Pi(ty, lam_ty_term)
            implicit_spine = (implicit,) + implicit_spine
        return lam_term, ElabType(lam_ty_term, implicit_spine)

    def elab_check(self, env: ElabEnv, state: "ElabState", expected: ElabType) -> Term:
        binder_tys: list[Term] = []
        binder_impls: list[bool] = []
        env1 = env
        expected_ty = expected
        for binder in self.binders:
            pi_ty = expected_ty.term.whnf(env1.kenv)
            if not isinstance(pi_ty, Pi):
                raise SurfaceError("Lambda needs expected function type", self.span)
            if binder.ty is None:
                binder_ty = pi_ty.arg_ty
            else:
                binder_ty, binder_ty_ty = binder.ty.elab_infer(env1, state)
                binder_ty_ty_whnf = binder_ty_ty.term.whnf(env1.kenv)
                if not isinstance(binder_ty_ty_whnf, Univ):
                    raise SurfaceError("Binder type must be a universe", binder.span)
                state.add_constraint(env1.kenv, binder_ty, pi_ty.arg_ty, binder.span)
            binder_tys.append(binder_ty)
            binder_impls.append(binder.implicit)
            env1 = env1.push_binder(
                ElabType(binder_ty, _implicit_spine(binder.ty)), name=binder.name
            )
            expected_ty = ElabType(pi_ty.return_ty, expected_ty.implicit_spine[1:])
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

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        binder_tys, binder_impls, env1 = _elab_binders(env, state, self.binders)
        body_term, body_ty = self.body.elab_infer(env1, state)
        body_ty_whnf = body_ty.term.whnf(env1.kenv)
        if not isinstance(body_ty_whnf, Univ):
            raise SurfaceError("Pi body must be a universe", self.body.span)
        body_level = body_ty_whnf.level
        pi_term: Term = body_term
        result_level: LevelExpr | None = None
        for ty in reversed(binder_tys):
            arg_ty_ty = ty.infer_type(env.kenv).whnf(env.kenv)
            if not isinstance(arg_ty_ty, Univ):
                raise SurfaceError("Pi domain must be a universe", self.span)
            arg_level = arg_ty_ty.level
            if result_level is None:
                result_level = state.fresh_level_meta("type", self.span)
            state.add_level_constraint(arg_level, result_level, self.span)
            state.add_level_constraint(body_level, result_level, self.span)
            body_level = result_level
            pi_term = Pi(ty, pi_term)
        if result_level is None:
            result_level = state.fresh_level_meta("type", self.span)
            state.add_level_constraint(body_level, result_level, self.span)
        implicit_spine = tuple(binder_impls) + body_ty.implicit_spine
        return pi_term, ElabType(Univ(result_level), implicit_spine)

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
    args: tuple[SArg, ...]

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        fn_term, fn_ty = self.fn.elab_infer(env, state)
        implicit_spine = _implicit_spine_for_term(fn_term, env)
        spine_index = 0
        pending = list(self.args)
        while pending:
            fn_ty_whnf = fn_ty.term.whnf(env.kenv)
            if not isinstance(fn_ty_whnf, Pi):
                raise SurfaceError("Application of non-function", pending[0].term.span)
            binder_is_implicit = False
            if implicit_spine is not None and spine_index < len(implicit_spine):
                binder_is_implicit = implicit_spine[spine_index]
            arg = pending.pop(0)
            if binder_is_implicit and not arg.implicit:
                meta = state.fresh_meta(
                    env.kenv, fn_ty_whnf.arg_ty, self.span, kind="implicit"
                )
                fn_term = App(fn_term, meta)
                fn_ty = ElabType(
                    fn_ty_whnf.return_ty.subst(meta),
                    fn_ty.implicit_spine[1:],
                )
                spine_index += 1
                pending.insert(0, arg)
                continue
            if not binder_is_implicit and arg.implicit and implicit_spine is not None:
                raise SurfaceError(
                    "Implicit argument provided where explicit expected", arg.term.span
                )
            arg_term = arg.term.elab_check(env, state, ElabType(fn_ty_whnf.arg_ty))
            fn_term = App(fn_term, arg_term)
            fn_ty = ElabType(
                fn_ty_whnf.return_ty.subst(arg_term),
                fn_ty.implicit_spine[1:],
            )
            spine_index += 1
        while True:
            fn_ty_whnf = fn_ty.term.whnf(env.kenv)
            if not isinstance(fn_ty_whnf, Pi):
                break
            if implicit_spine is None or spine_index >= len(implicit_spine):
                break
            if not implicit_spine[spine_index]:
                break
            meta = state.fresh_meta(
                env.kenv, fn_ty_whnf.arg_ty, self.span, kind="implicit"
            )
            fn_term = App(fn_term, meta)
            fn_ty = ElabType(
                fn_ty_whnf.return_ty.subst(meta),
                fn_ty.implicit_spine[1:],
            )
            spine_index += 1
        return fn_term, fn_ty

    def resolve(self, env: Env, names: NameEnv) -> Term:
        fn = self.fn.resolve(env, names)
        term = fn
        for arg in self.args:
            arg_term = arg.term.resolve(env, names)
            term = App(term, arg_term)
        return term


@dataclass(frozen=True)
class SUApp(SurfaceTerm):
    head: SurfaceTerm
    levels: tuple[int, ...]

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
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
        return uapp, ElabType(uapp.infer_type(env.kenv))

    def resolve(self, env: Env, names: NameEnv) -> Term:
        from mltt.kernel.ast import UApp

        head = self.head.resolve(env, names)
        levels = tuple(LConst(level) for level in self.levels)
        return UApp(head, levels)


@dataclass(frozen=True)
class SLet(SurfaceTerm):
    uparams: tuple[str, ...]
    name: str
    ty: SurfaceTerm
    val: SurfaceTerm
    body: SurfaceTerm

    def elab_infer(self, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
        if len(set(self.uparams)) != len(self.uparams):
            raise SurfaceError("Duplicate universe binder", self.span)
        old_level_names = state.level_names
        state.level_names = list(reversed(self.uparams)) + state.level_names
        ty_term, ty_ty = self.ty.elab_infer(env, state)
        ty_ty_whnf = ty_ty.term.whnf(env.kenv)
        if not isinstance(ty_ty_whnf, Univ):
            raise SurfaceError("Let type must be a universe", self.span)
        val_term = self.val.elab_check(env, state, ElabType(ty_term))
        state.level_names = old_level_names
        uarity, ty_term, val_term = state.generalize_levels_for_let(ty_term, val_term)
        implicit_spine = _implicit_spine(self.ty)
        env1 = env.push_let(
            ElabType(ty_term, implicit_spine),
            val_term,
            name=self.name,
            uarity=uarity,
        )
        env1.eglobals[self.name] = ElabType(ty_term, implicit_spine)
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
    env: ElabEnv, state: "ElabState", binders: tuple[SBinder, ...]
) -> tuple[list[Term], list[bool], ElabEnv]:
    binder_tys: list[Term] = []
    binder_impls: list[bool] = []
    for binder in binders:
        ty_term, env = binder.elab(env, state)
        binder_tys.append(ty_term)
        binder_impls.append(binder.implicit)
    return binder_tys, binder_impls, env


def _implicit_spine(term: SurfaceTerm | None) -> tuple[bool, ...]:
    if term is None:
        return ()
    spine: list[bool] = []
    current = term
    while isinstance(current, SPi):
        spine.extend(b.implicit for b in current.binders)
        current = current.body
    return tuple(spine)


def _implicit_spine_for_term(term: Term, env: ElabEnv) -> tuple[bool, ...] | None:
    head = term
    applied = 0
    while isinstance(head, App):
        applied += 1
        head = head.func
    if isinstance(head, UApp):
        head = head.head
    if isinstance(head, Var):
        implicit_spine = env.locals[head.k].implicit_spine
    elif isinstance(head, Const):
        gty = env.global_type(head.name)
        assert gty is not None
        implicit_spine = gty.implicit_spine
    else:
        return None
    if applied >= len(implicit_spine):
        return ()
    return implicit_spine[applied:]
