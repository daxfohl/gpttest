"""Surface AST and elaboration helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from mltt.kernel.ast import App, Lam, Let, Pi, Term, Univ, Var, UApp
from mltt.kernel.env import Const, Env, GlobalDecl
from mltt.kernel.ind import Ctor, Ind
from mltt.kernel.levels import LConst, LevelExpr
from mltt.kernel.tel import ArgList
from mltt.surface.etype import ElabEnv, ElabType

if TYPE_CHECKING:
    from mltt.surface.elab_state import ElabState
    from mltt.surface.match import PatCtor


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
        term_k, term_ty = self.elab_infer(env, state)
        state.add_constraint(env.kenv, term_ty.term, expected.term, self.span)
        return term_k

    def resolve(self, env: Env, names: NameEnv) -> Term:
        raise SurfaceError("Unsupported surface term", self.span)


def _expect_universe(term: Term, env: Env, span: Span) -> Univ:
    ty_whnf = term.whnf(env)
    if not isinstance(ty_whnf, Univ):
        raise SurfaceError(f"{type(term).__name__} must be a universe", span)
    return ty_whnf


def _require_global_info(
    env: ElabEnv, name: str, span: Span, message: str
) -> tuple[GlobalDecl, ElabType]:
    info = env.global_info(name)
    if info is None:
        raise SurfaceError(message, span)
    return info


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
        _expect_universe(ty_ty.term, env.kenv, self.span)
        implicit_spine = _implicit_spine(self.ty)
        return ty_term, env.push_binder(
            ElabType(ty_term, implicit_spine), name=self.name
        )


@dataclass(frozen=True)
class SArg:
    term: SurfaceTerm
    implicit: bool = False
    name: str | None = None


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
            term, levels = state.apply_implicit_levels(
                Var(idx), binder.uarity, self.span
            )
            ty = env.local_type(idx).inst_levels(levels)
            return term, ElabType(
                state.zonk(ty.term), ty.implicit_spine, ty.binder_names
            )
        decl, gty = _require_global_info(
            env, self.name, self.span, f"Unknown identifier {self.name}"
        )
        head: Term
        if isinstance(decl.value, (Ind, Ctor)):
            head = decl.value
        else:
            head = Const(self.name)
        term, levels = state.apply_implicit_levels(head, decl.uarity, self.span)
        ty = gty.inst_levels(levels)
        return term, ElabType(state.zonk(ty.term), ty.implicit_spine, ty.binder_names)

    def resolve(self, env: Env, names: NameEnv) -> Term:
        idx = names.lookup(self.name)
        if idx is not None:
            return Var(idx)
        decl = env.lookup_global(self.name)
        if decl is not None:
            if isinstance(decl.value, (Ind, Ctor)):
                return decl.value
            return Const(self.name)
        raise SurfaceError(f"Unknown identifier {self.name}", self.span)


@dataclass(frozen=True)
class SConst(SurfaceTerm):
    name: str

    def elab_infer(self, env: ElabEnv, state: "ElabState") -> tuple[Term, ElabType]:
        decl, gty = _require_global_info(
            env, self.name, self.span, f"Unknown constant {self.name}"
        )
        term, levels = state.apply_implicit_levels(
            Const(self.name), decl.uarity, self.span
        )
        ty = gty.inst_levels(levels)
        return term, ElabType(state.zonk(ty.term), ty.implicit_spine, ty.binder_names)

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
        _expect_universe(ty_ty.term, env.kenv, self.span)
        term = self.term.elab_check(env, state, ElabType(ty_term))
        return term, ElabType(ty_term, _implicit_spine(self.ty), _binder_names(self.ty))

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
        binder_tys, binder_impls, _binder_levels, env1 = _elab_binders(
            env, state, self.binders
        )
        body_term, body_ty = self.body.elab_infer(env1, state)
        lam_term = body_term
        lam_ty_term = body_ty.term
        implicit_spine = body_ty.implicit_spine
        binder_names = body_ty.binder_names
        for ty, implicit, name in reversed(
            list(zip(binder_tys, binder_impls, (b.name for b in self.binders)))
        ):
            lam_term = Lam(ty, lam_term)
            lam_ty_term = Pi(ty, lam_ty_term)
            implicit_spine = (implicit,) + implicit_spine
            binder_names = (name,) + binder_names
        return lam_term, ElabType(lam_ty_term, implicit_spine, binder_names)

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
                _expect_universe(binder_ty_ty.term, env1.kenv, binder.span)
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
        binder_tys, binder_impls, binder_levels, env1 = _elab_binders(
            env, state, self.binders
        )
        body_term, body_ty = self.body.elab_infer(env1, state)
        body_ty_whnf = _expect_universe(body_ty.term, env1.kenv, self.body.span)
        body_level = body_ty_whnf.level
        pi_term: Term = body_term
        result_level: LevelExpr | None = None
        for ty, arg_level in reversed(list(zip(binder_tys, binder_levels))):
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
        binder_names = fn_ty.binder_names
        spine_index = 0
        pending = list(self.args)
        applied_args: list[Term] = []
        applied_arg_types: list[Term] = []
        applied_arg_names: list[str | None] = []
        context_args: list[Term] = []
        context_arg_types: list[Term] = []
        context_arg_names: list[str | None] = []
        use_context = any(arg.name is not None for arg in pending)
        named_seen = False
        for item in pending:
            if item.name is not None:
                named_seen = True
                continue
            if named_seen:
                raise SurfaceError(
                    "Positional arguments must come before named arguments",
                    item.term.span,
                )
        if any(arg.name is not None for arg in pending) and not binder_names:
            raise SurfaceError("Named arguments require binder names", self.span)
        positional = [arg for arg in pending if arg.name is None]
        named: dict[str, SArg] = {}
        for item in pending:
            if item.name is None:
                continue
            if item.name in named:
                raise SurfaceError(
                    f"Duplicate named argument {item.name}", item.term.span
                )
            named[item.name] = item
        pos_index = 0
        while True:
            fn_ty_whnf = fn_ty.term.whnf(env.kenv)
            if not isinstance(fn_ty_whnf, Pi):
                if pos_index < len(positional) or named:
                    raise SurfaceError(
                        "Application of non-function",
                        (
                            positional[pos_index].term.span
                            if pos_index < len(positional)
                            else next(iter(named.values())).term.span
                        ),
                    )
                break
            binder_is_implicit = False
            if implicit_spine is not None and spine_index < len(implicit_spine):
                binder_is_implicit = implicit_spine[spine_index]
            binder_name = (
                binder_names[spine_index] if spine_index < len(binder_names) else None
            )
            arg: SArg | None = None
            consume_positional = False
            if binder_name is not None and binder_name in named:
                arg = named.pop(binder_name)
            elif pos_index < len(positional):
                candidate = positional[pos_index]
                if binder_is_implicit and candidate.implicit:
                    arg = candidate
                    consume_positional = True
                elif not binder_is_implicit:
                    arg = candidate
                    consume_positional = True
            if arg is None:
                if binder_is_implicit:
                    meta = state.fresh_meta(
                        env.kenv, fn_ty_whnf.arg_ty, self.span, kind="implicit"
                    )
                    fn_term = App(fn_term, meta)
                    fn_ty = ElabType(
                        fn_ty_whnf.return_ty.subst(meta),
                        fn_ty.implicit_spine[1:],
                        fn_ty.binder_names[1:],
                    )
                    applied_args.append(meta)
                    applied_arg_types.append(fn_ty_whnf.arg_ty)
                    applied_arg_names.append(binder_name)
                    spine_index += 1
                    continue
                raise SurfaceError("Missing explicit argument", self.span)
            if not binder_is_implicit and arg.implicit and implicit_spine is not None:
                raise SurfaceError(
                    "Implicit argument provided where explicit expected", arg.term.span
                )
            arg_env = env
            arg_ty = fn_ty_whnf.arg_ty
            if use_context and context_args:
                allow_names = arg.name is not None
                for prev_ty, prev_name in zip(
                    context_arg_types, context_arg_names, strict=True
                ):
                    name = prev_name if allow_names else None
                    arg_env = arg_env.push_binder(ElabType(prev_ty), name=name)
                arg_ty = arg_ty.shift(len(context_args))
            arg_term = arg.term.elab_check(arg_env, state, ElabType(arg_ty))
            if use_context and context_args:
                arg_term = arg_term.instantiate(ArgList.of(*context_args))
            fn_term = App(fn_term, arg_term)
            fn_ty = ElabType(
                fn_ty_whnf.return_ty.subst(arg_term),
                fn_ty.implicit_spine[1:],
                fn_ty.binder_names[1:],
            )
            applied_args.append(arg_term)
            applied_arg_types.append(fn_ty_whnf.arg_ty)
            applied_arg_names.append(binder_name)
            context_args.append(arg_term)
            context_arg_types.append(fn_ty_whnf.arg_ty)
            context_arg_names.append(binder_name)
            spine_index += 1
            if consume_positional:
                pos_index += 1
        if named:
            unknown = next(iter(named.keys()))
            raise SurfaceError(f"Unknown named argument {unknown}", self.span)
        while True:
            fn_ty_whnf = fn_ty.term.whnf(env.kenv)
            if not isinstance(fn_ty_whnf, Pi):
                break
            if implicit_spine is None or spine_index >= len(implicit_spine):
                break
            if not implicit_spine[spine_index]:
                raise SurfaceError("Missing explicit argument", self.span)
            meta = state.fresh_meta(
                env.kenv, fn_ty_whnf.arg_ty, self.span, kind="implicit"
            )
            fn_term = App(fn_term, meta)
            fn_ty = ElabType(
                fn_ty_whnf.return_ty.subst(meta),
                fn_ty.implicit_spine[1:],
                fn_ty.binder_names[1:],
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
                if isinstance(decl.value, (Ind, Ctor)):
                    head_term = decl.value
                else:
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
                if isinstance(head_term, UApp) and isinstance(head_term.head, Ind):
                    head_term = head_term.head
            case SCtor(name=name):
                decl = env.lookup_global(name)
                if decl is None or decl.value is None:
                    raise SurfaceError(f"Unknown constructor {name}", self.span)
                head_term = decl.value
                if isinstance(head_term, UApp) and isinstance(head_term.head, Ctor):
                    head_term = head_term.head
            case _:
                head_term, _ = self.head.elab_infer(env, state)
                if isinstance(head_term, UApp):
                    head_term = head_term.head
        level_terms: tuple[LevelExpr, ...] = tuple(
            (
                LConst(level)
                if isinstance(level, int)
                else state.lookup_level(level, self.span)
            )
            for level in self.levels
        )
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
        _expect_universe(ty_ty.term, env.kenv, self.span)
        val_term = _desugar_equation_rec(self.name, self.val).elab_check(
            env, state, ElabType(ty_term)
        )
        if not self.uparams:
            ty_term, val_term = state.merge_type_level_metas([ty_term, val_term])
        state.level_names = old_level_names
        uarity, ty_term, val_term = state.generalize_levels_for_let(ty_term, val_term)
        implicit_spine = _implicit_spine(self.ty)
        binder_names = _binder_names(self.ty)
        env1 = env.push_let(
            ElabType(ty_term, implicit_spine, binder_names),
            val_term,
            name=self.name,
            uarity=uarity,
        )
        env1.eglobals[self.name] = ElabType(ty_term, implicit_spine, binder_names)
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
) -> tuple[list[Term], list[bool], list[LevelExpr], ElabEnv]:
    binder_tys: list[Term] = []
    binder_impls: list[bool] = []
    binder_levels: list[LevelExpr] = []
    for binder in binders:
        if binder.ty is None:
            raise SurfaceError("Missing binder type", binder.span)
        ty_term, ty_ty = binder.ty.elab_infer(env, state)
        ty_ty_whnf = _expect_universe(ty_ty.term, env.kenv, binder.span)
        implicit_spine = _implicit_spine(binder.ty)
        binder_names = _binder_names(binder.ty)
        env = env.push_binder(
            ElabType(ty_term, implicit_spine, binder_names), name=binder.name
        )
        binder_tys.append(ty_term)
        binder_impls.append(binder.implicit)
        binder_levels.append(ty_ty_whnf.level)
    return binder_tys, binder_impls, binder_levels, env


def _implicit_spine(term: SurfaceTerm | None) -> tuple[bool, ...]:
    if term is None:
        return ()
    spine: list[bool] = []
    current = term
    while isinstance(current, SPi):
        spine.extend(b.implicit for b in current.binders)
        current = current.body
    return tuple(spine)


def _binder_names(term: SurfaceTerm | None) -> tuple[str | None, ...]:
    if term is None:
        return ()
    names: list[str | None] = []
    current = term
    while isinstance(current, SPi):
        names.extend(b.name for b in current.binders)
        current = current.body
    return tuple(names)


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
    elif isinstance(head, Ind):
        gty = env.global_type(head.name)
        if gty is None:
            return None
        implicit_spine = gty.implicit_spine
    elif isinstance(head, Ctor):
        ctor_name = f"{head.inductive.name}.{head.name}"
        gty = env.global_type(ctor_name)
        if gty is None:
            return None
        implicit_spine = gty.implicit_spine
    else:
        return None
    if applied >= len(implicit_spine):
        return ()
    return implicit_spine[applied:]


def _desugar_equation_rec(name: str, term: SurfaceTerm) -> SurfaceTerm:
    from mltt.surface.match import PatCtor, PatVar, SBranch, SMatch

    if not isinstance(term, SLam):
        return term
    binder_names = [binder.name for binder in term.binders]
    if not binder_names:
        return term
    match_term = None
    match_args: list[SArg] = []
    if isinstance(term.body, SMatch):
        match_term = term.body
    else:
        head, args = _decompose_sapp(term.body)
        if isinstance(head, SMatch):
            match_term = head
            match_args = args
    if match_term is None:
        return term
    if len(match_term.scrutinees) != 1:
        return term
    scrutinee = match_term.scrutinees[0]
    if not isinstance(scrutinee, SVar):
        return term
    if scrutinee.name not in binder_names:
        return term
    scrut_index = binder_names.index(scrutinee.name)
    used_any = False
    new_branches: list[SBranch] = []
    for branch in match_term.branches:
        pat = branch.pat
        if not isinstance(pat, PatCtor):
            new_branches.append(branch)
            continue
        ih_name = _fresh_ih_name(binder_names, pat)
        new_rhs, used_var = _replace_recursive_call(
            name,
            branch.rhs,
            binder_names,
            scrut_index,
            ih_name,
            match_args,
        )
        if used_var is None:
            new_branches.append(branch)
            continue
        used_any = True
        new_args = pat.args + (PatVar(name=ih_name, span=pat.span),)
        new_pat = PatCtor(span=pat.span, ctor=pat.ctor, args=new_args)
        new_branches.append(SBranch(pat=new_pat, rhs=new_rhs, span=branch.span))
    if not used_any:
        return term
    new_match = SMatch(
        span=match_term.span,
        scrutinees=match_term.scrutinees,
        as_names=match_term.as_names,
        motive=match_term.motive,
        branches=tuple(new_branches),
    )
    if match_args:
        rebuilt = SApp(span=term.body.span, fn=new_match, args=tuple(match_args))
        return SLam(span=term.span, binders=term.binders, body=rebuilt)
    return SLam(span=term.span, binders=term.binders, body=new_match)


def _fresh_ih_name(binder_names: list[str], pat: "PatCtor") -> str:
    from mltt.surface.match import PatVar

    existing = {name for name in binder_names if name != "_"}
    for arg in pat.args:
        if isinstance(arg, PatVar):
            existing.add(arg.name)
    if "ih" not in existing:
        return "ih"
    index = 1
    while f"ih{index}" in existing:
        index += 1
    return f"ih{index}"


def _replace_recursive_call(
    name: str,
    term: SurfaceTerm,
    binder_names: list[str],
    scrut_index: int,
    ih_name: str,
    ih_args: list[SArg],
) -> tuple[SurfaceTerm, str | None]:
    used_var: str | None = None

    def decompose_app(app: SurfaceTerm) -> tuple[SurfaceTerm, list[SArg]]:
        if isinstance(app, SApp):
            head, args = decompose_app(app.fn)
            return head, args + list(app.args)
        return app, []

    def is_recursive_call(t: SurfaceTerm) -> str | None:
        head, args = decompose_app(t)
        if not isinstance(head, SVar) or head.name != name or not args:
            return None
        if any(arg.implicit for arg in args):
            return None
        if len(args) != len(binder_names):
            return None
        candidate: str | None = None
        for idx, (arg, binder) in enumerate(zip(args, binder_names, strict=True)):
            if idx == scrut_index:
                if not isinstance(arg.term, SVar):
                    return None
                candidate = arg.term.name
            else:
                if not isinstance(arg.term, SVar) or arg.term.name != binder:
                    return None
        return candidate

    def replace(t: SurfaceTerm) -> SurfaceTerm:
        nonlocal used_var
        candidate = is_recursive_call(t)
        if candidate is not None:
            if used_var is None:
                used_var = candidate
            elif used_var != candidate:
                raise SurfaceError(
                    "Recursive call uses multiple scrutinee vars", t.span
                )
            if ih_args:
                return SApp(
                    span=t.span,
                    fn=SVar(span=t.span, name=ih_name),
                    args=tuple(
                        SArg(term=arg.term, implicit=arg.implicit, name=arg.name)
                        for arg in ih_args
                    ),
                )
            return SVar(span=t.span, name=ih_name)
        if isinstance(t, SApp):
            new_fn = replace(t.fn)
            new_args: list[SArg] = []
            changed = new_fn is not t.fn
            for arg in t.args:
                new_term = replace(arg.term)
                changed = changed or new_term is not arg.term
                new_args.append(SArg(new_term, implicit=arg.implicit, name=arg.name))
            if changed:
                return SApp(span=t.span, fn=new_fn, args=tuple(new_args))
            return t
        if isinstance(t, SLam):
            new_body = replace(t.body)
            if new_body is not t.body:
                return SLam(span=t.span, binders=t.binders, body=new_body)
            return t
        if isinstance(t, SPi):
            new_body = replace(t.body)
            if new_body is not t.body:
                return SPi(span=t.span, binders=t.binders, body=new_body)
            return t
        if isinstance(t, SLet):
            new_val = replace(t.val)
            new_body = replace(t.body)
            if new_val is not t.val or new_body is not t.body:
                return SLet(
                    span=t.span,
                    uparams=t.uparams,
                    name=t.name,
                    ty=t.ty,
                    val=new_val,
                    body=new_body,
                )
            return t
        if isinstance(t, SAnn):
            new_term = replace(t.term)
            if new_term is not t.term:
                return SAnn(span=t.span, term=new_term, ty=t.ty)
            return t
        if isinstance(t, SUApp):
            new_head = replace(t.head)
            if new_head is not t.head:
                return SUApp(span=t.span, head=new_head, levels=t.levels)
            return t
        if "SMatch" in type(t).__name__:
            from mltt.surface.match import SBranch as MatchBranch
            from mltt.surface.match import SMatch

            if isinstance(t, SMatch):
                new_scrutinees = tuple(replace(s) for s in t.scrutinees)
                new_branches = []
                changed = new_scrutinees != t.scrutinees
                for br in t.branches:
                    new_rhs = replace(br.rhs)
                    changed = changed or new_rhs is not br.rhs
                    new_branches.append(MatchBranch(br.pat, new_rhs, br.span))
                new_motive = replace(t.motive) if t.motive is not None else None
                changed = changed or new_motive is not t.motive
                if changed:
                    return SMatch(
                        span=t.span,
                        scrutinees=new_scrutinees,
                        as_names=t.as_names,
                        motive=new_motive,
                        branches=tuple(new_branches),
                    )
            return t
        return t

    replaced = replace(term)
    return replaced, used_var


def _decompose_sapp(term: SurfaceTerm) -> tuple[SurfaceTerm, list[SArg]]:
    if isinstance(term, SApp):
        head, args = _decompose_sapp(term.fn)
        return head, args + list(term.args)
    return term, []
