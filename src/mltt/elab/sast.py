"""Surface elaboration helpers."""

from __future__ import annotations

from mltt.kernel.ast import App, Lam, Let, Pi, Term, Univ, Var, UApp
from mltt.kernel.env import Const, Env, GlobalDecl
from mltt.kernel.ind import Ctor, Ind
from mltt.kernel.levels import LConst, LevelExpr
from mltt.elab.elab_state import ElabState
from mltt.elab.etype import ElabEnv, ElabType
from mltt.elab.names import NameEnv
from mltt.surface.sast import (
    SAnn,
    SApp,
    SArg,
    SBinder,
    SConst,
    SCtor,
    SHole,
    SInd,
    SInductiveDef,
    SLet,
    SLetPat,
    SLam,
    SMatch,
    SPi,
    SPartial,
    SUApp,
    SUniv,
    SVar,
    Span,
    SurfaceError,
    SurfaceTerm,
)


def elab_infer(
    term: SurfaceTerm, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    match term:
        case SVar(name=name):
            ctx_term = env.lookup_context_term(name)
            if ctx_term is not None:
                term_k, ty = ctx_term
                return term_k, ElabType(
                    state.zonk(ty.term), ty.implicit_spine, ty.binder_names
                )
            idx = env.lookup_local(name)
            if idx is not None:
                binder = env.binders[idx]
                term_k, levels = state.apply_implicit_levels(
                    Var(idx), binder.uarity, term.span
                )
                ty = env.local_type(idx).inst_levels(levels)
                return term_k, ElabType(
                    state.zonk(ty.term), ty.implicit_spine, ty.binder_names
                )
            decl, gty = _require_global_info(
                env, name, term.span, f"Unknown identifier {name}"
            )
            head: Term
            if isinstance(decl.value, (Ind, Ctor)):
                head = decl.value
            else:
                head = Const(name)
            term_k, levels = state.apply_implicit_levels(head, decl.uarity, term.span)
            ty = gty.inst_levels(levels)
            return term_k, ElabType(
                state.zonk(ty.term), ty.implicit_spine, ty.binder_names
            )
        case SConst(name=name):
            decl, gty = _require_global_info(
                env, name, term.span, f"Unknown constant {name}"
            )
            term_k, levels = state.apply_implicit_levels(
                Const(name), decl.uarity, term.span
            )
            ty = gty.inst_levels(levels)
            return term_k, ElabType(
                state.zonk(ty.term), ty.implicit_spine, ty.binder_names
            )
        case SUniv(level=level):
            level_expr: LevelExpr | int
            if level is None:
                level_expr = state.fresh_level_meta("type", term.span)
            elif isinstance(level, str):
                level_expr = state.lookup_level(level, term.span)
            else:
                level_expr = level
            term_k = Univ(level_expr)
            return term_k, ElabType(Univ(LevelExpr.of(level_expr).succ()))
        case SAnn(term=inner, ty=ty_src):
            ty_term, ty_ty = elab_infer(ty_src, env, state)
            _expect_universe(ty_ty.term, env.kenv, term.span)
            term_k = elab_check(inner, env, state, ElabType(ty_term))
            return term_k, ElabType(ty_term)
        case SHole():
            raise SurfaceError("Hole needs expected type", term.span)
        case SLam():
            return _elab_lam_infer(term, env, state)
        case SPi():
            return _elab_pi_infer(term, env, state)
        case SApp(fn=fn, args=args):
            return _elab_apply(fn, args, env, state, term.span, allow_partial=False)
        case SUApp():
            return _elab_uapp_infer(term, env, state)
        case SPartial(term=inner):
            return _elab_partial_infer(inner, env, state, term.span)
        case SLet():
            return _elab_let_infer(term, env, state)
        case SMatch():
            from mltt.elab.match import elab_match_infer

            return elab_match_infer(term, env, state)
        case SLetPat():
            from mltt.elab.match import elab_let_pat_infer

            return elab_let_pat_infer(term, env, state)
        case SInd():
            from mltt.elab.sind import elab_ind_infer

            return elab_ind_infer(term, env, state)
        case SCtor():
            from mltt.elab.sind import elab_ctor_infer

            return elab_ctor_infer(term, env, state)
        case SInductiveDef():
            from mltt.elab.sind import elab_inductive_infer

            return elab_inductive_infer(term, env, state)
        case _:
            raise SurfaceError("Unsupported surface term", term.span)


def elab_check(
    term: SurfaceTerm, env: ElabEnv, state: ElabState, expected: ElabType
) -> Term:
    match term:
        case SHole():
            return state.fresh_meta(env.kenv, expected.term, term.span, kind="hole")
        case SLam():
            return _elab_lam_check(term, env, state, expected)
        case SMatch():
            from mltt.elab.match import elab_match_check

            return elab_match_check(term, env, state, expected)
        case SAnn(term=inner, ty=ty_src):
            ty_term, ty_ty = elab_infer(ty_src, env, state)
            _expect_universe(ty_ty.term, env.kenv, term.span)
            term_k = elab_check(inner, env, state, ElabType(ty_term))
            state.add_constraint(env.kenv, ty_term, expected.term, term.span)
            return term_k
        case _:
            term_k, term_ty = elab_infer(term, env, state)
            state.add_constraint(env.kenv, term_ty.term, expected.term, term.span)
            return term_k


def resolve(term: SurfaceTerm, env: Env, names: NameEnv) -> Term:
    match term:
        case SVar(name=name):
            idx = names.lookup(name)
            if idx is not None:
                return Var(idx)
            decl = env.lookup_global(name)
            if decl is not None:
                if isinstance(decl.value, (Ind, Ctor)):
                    return decl.value
                return Const(name)
            raise SurfaceError(f"Unknown identifier {name}", term.span)
        case SConst(name=name):
            if env.lookup_global(name) is None:
                raise SurfaceError(f"Unknown constant {name}", term.span)
            return Const(name)
        case SUniv(level=level):
            if level is None:
                raise SurfaceError("Universe requires elaboration", term.span)
            if isinstance(level, str):
                raise SurfaceError("Universe requires elaboration", term.span)
            return Univ(level)
        case SAnn(term=inner, ty=ty_src):
            _ = ty_src
            return resolve(inner, env, names)
        case SHole():
            raise SurfaceError("Hole requires elaboration", term.span)
        case SLam(binders=binders, body=body):
            if any(b.ty is None for b in binders):
                raise SurfaceError("Missing binder type", term.span)
            tys: list[Term] = []
            for binder in binders:
                assert binder.ty is not None
                tys.append(resolve(binder.ty, env, names))
                names.push(binder.name)
            term_k = resolve(body, env, names)
            for ty in reversed(tys):
                term_k = Lam(ty, term_k)
                names.pop()
            return term_k
        case SPi(binders=binders, body=body):
            if any(b.ty is None for b in binders):
                raise SurfaceError("Missing binder type", term.span)
            pi_tys: list[Term] = []
            for binder in binders:
                assert binder.ty is not None
                pi_tys.append(resolve(binder.ty, env, names))
                names.push(binder.name)
            term_k = resolve(body, env, names)
            for ty in reversed(pi_tys):
                term_k = Pi(ty, term_k)
                names.pop()
            return term_k
        case SApp(fn=fn, args=args):
            term_k = resolve(fn, env, names)
            for arg in args:
                arg_term = resolve(arg.term, env, names)
                term_k = App(term_k, arg_term)
            return term_k
        case SUApp(head=head, levels=levels):
            head_term = resolve(head, env, names)
            if any(isinstance(level, str) for level in levels):
                raise SurfaceError("Universe requires elaboration", term.span)
            int_levels: list[int] = []
            for level in levels:
                if isinstance(level, int):
                    int_levels.append(level)
                else:
                    raise SurfaceError("Universe requires elaboration", term.span)
            level_terms = tuple(LConst(level) for level in int_levels)
            return UApp(head_term, level_terms)
        case SPartial(term=inner):
            return resolve(inner, env, names)
        case SLet(name=name, ty=ty_src, val=val_src, body=body):
            if ty_src is None:
                raise SurfaceError("Missing let type; needs elaboration", term.span)
            ty_term = resolve(ty_src, env, names)
            val_term = resolve(val_src, env, names)
            names.push(name)
            body_term = resolve(body, env, names)
            names.pop()
            return Let(ty_term, val_term, body_term)
        case SMatch() | SLetPat() | SInd() | SCtor() | SInductiveDef():
            raise SurfaceError("Surface construct requires elaboration", term.span)
        case _:
            raise SurfaceError("Unsupported surface term", term.span)


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


def _elab_lam_infer(
    term: SLam, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    if any(b.ty is None for b in term.binders):
        raise SurfaceError(
            "Cannot infer unannotated lambda; add binder types or use check-mode",
            term.span,
        )
    binder_tys, binder_impls, _binder_levels, env1 = _elab_binders(
        env, state, term.binders
    )
    body_term, body_ty = elab_infer(term.body, env1, state)
    lam_term = body_term
    lam_ty_term = body_ty.term
    implicit_spine = body_ty.implicit_spine
    binder_names = body_ty.binder_names
    for ty, implicit, name in reversed(
        list(zip(binder_tys, binder_impls, (b.name for b in term.binders)))
    ):
        lam_term = Lam(ty, lam_term)
        lam_ty_term = Pi(ty, lam_ty_term)
        implicit_spine = (implicit,) + implicit_spine
        binder_names = (name,) + binder_names
    return lam_term, ElabType(lam_ty_term, implicit_spine, binder_names)


def _elab_lam_check(
    term: SLam, env: ElabEnv, state: ElabState, expected: ElabType
) -> Term:
    binder_tys: list[Term] = []
    binder_impls: list[bool] = []
    env1 = env
    expected_ty = expected
    for binder in term.binders:
        pi_ty = expected_ty.term.whnf(env1.kenv)
        if not isinstance(pi_ty, Pi):
            raise SurfaceError("Lambda needs expected function type", term.span)
        if binder.ty is None:
            binder_ty = pi_ty.arg_ty
        else:
            binder_ty, binder_ty_ty = elab_infer(binder.ty, env1, state)
            _expect_universe(binder_ty_ty.term, env1.kenv, binder.span)
            state.add_constraint(env1.kenv, binder_ty, pi_ty.arg_ty, binder.span)
        binder_tys.append(binder_ty)
        binder_impls.append(binder.implicit)
        env1 = env1.push_binder(
            ElabType(binder_ty, _implicit_spine(binder.ty)), name=binder.name
        )
        expected_ty = ElabType(pi_ty.return_ty, expected_ty.implicit_spine[1:])
    body_term = elab_check(term.body, env1, state, expected_ty)
    lam_term = body_term
    for ty in reversed(binder_tys):
        lam_term = Lam(ty, lam_term)
    return lam_term


def _elab_pi_infer(term: SPi, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
    binder_tys, binder_impls, binder_levels, env1 = _elab_binders(
        env, state, term.binders
    )
    body_term, body_ty = elab_infer(term.body, env1, state)
    body_ty_whnf = _expect_universe(body_ty.term, env1.kenv, term.body.span)
    body_level = body_ty_whnf.level
    pi_term: Term = body_term
    result_level: LevelExpr | None = None
    for ty, arg_level in reversed(list(zip(binder_tys, binder_levels))):
        if result_level is None:
            result_level = state.fresh_level_meta("type", term.span)
        state.add_level_constraint(arg_level, result_level, term.span)
        state.add_level_constraint(body_level, result_level, term.span)
        body_level = result_level
        pi_term = Pi(ty, pi_term)
    if result_level is None:
        result_level = state.fresh_level_meta("type", term.span)
        state.add_level_constraint(body_level, result_level, term.span)
    implicit_spine = tuple(binder_impls) + body_ty.implicit_spine
    return pi_term, ElabType(Univ(result_level), implicit_spine)


def _elab_uapp_infer(
    term: SUApp, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    head_term: Term
    match term.head:
        case SVar(name=name):
            if env.lookup_local(name) is not None:
                raise SurfaceError("UApp head must be a global", term.span)
            decl = env.lookup_global(name)
            if decl is None:
                raise SurfaceError(f"Unknown identifier {name}", term.span)
            if isinstance(decl.value, (Ind, Ctor)):
                head_term = decl.value
            else:
                head_term = Const(name)
        case SConst(name=name):
            decl = env.lookup_global(name)
            if decl is None:
                raise SurfaceError(f"Unknown constant {name}", term.span)
            head_term = Const(name)
        case SInd(name=name):
            decl = env.lookup_global(name)
            if decl is None or decl.value is None:
                raise SurfaceError(f"Unknown inductive {name}", term.span)
            head_term = decl.value
            if isinstance(head_term, UApp) and isinstance(head_term.head, Ind):
                head_term = head_term.head
        case SCtor(name=name):
            decl = env.lookup_global(name)
            if decl is None or decl.value is None:
                raise SurfaceError(f"Unknown constructor {name}", term.span)
            head_term = decl.value
            if isinstance(head_term, UApp) and isinstance(head_term.head, Ctor):
                head_term = head_term.head
        case _:
            head_term, _ = elab_infer(term.head, env, state)
            if isinstance(head_term, UApp):
                head_term = head_term.head
    level_terms: tuple[LevelExpr, ...] = tuple(
        (
            LConst(level)
            if isinstance(level, int)
            else state.lookup_level(level, term.span)
        )
        for level in term.levels
    )
    uapp = UApp(head_term, level_terms)
    return uapp, ElabType(uapp.infer_type(env.kenv))


def _elab_partial_infer(
    term: SurfaceTerm, env: ElabEnv, state: ElabState, span: Span
) -> tuple[Term, ElabType]:
    match term:
        case SApp(fn=fn, args=args):
            return _elab_apply(fn, args, env, state, span, allow_partial=True)
    return elab_infer(term, env, state)


def _elab_let_infer(
    term: SLet, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    if len(set(term.uparams)) != len(term.uparams):
        raise SurfaceError("Duplicate universe binder", term.span)
    old_level_names = state.level_names
    state.level_names = list(reversed(term.uparams)) + state.level_names
    val_src = term.val
    if term.ty is None:
        val_term, val_ty = elab_infer(val_src, env, state)
        ty_term = val_ty.term
        implicit_spine = val_ty.implicit_spine
        binder_names = val_ty.binder_names
        if not term.uparams:
            ty_term, val_term = state.merge_type_level_metas([ty_term, val_term])
    else:
        ty_term, ty_ty = elab_infer(term.ty, env, state)
        _expect_universe(ty_ty.term, env.kenv, term.span)
        implicit_spine = _implicit_spine(term.ty)
        binder_names = _binder_names(term.ty)
        if not any(binder_names):
            try:
                val_term, val_ty = elab_infer(val_src, env, state)
            except SurfaceError:
                val_term = elab_check(val_src, env, state, ElabType(ty_term))
            else:
                state.add_constraint(env.kenv, val_ty.term, ty_term, term.span)
                binder_names = val_ty.binder_names
        else:
            val_term = elab_check(val_src, env, state, ElabType(ty_term))
        if not term.uparams:
            ty_term, val_term = state.merge_type_level_metas([ty_term, val_term])
    state.level_names = old_level_names
    uarity, ty_term, val_term = state.generalize_levels_for_let(ty_term, val_term)
    if not any(binder_names):
        binder_names = _binder_names_from_term(term.val)
    env1 = env.push_let(
        ElabType(ty_term, implicit_spine, binder_names),
        val_term,
        name=term.name,
        uarity=uarity,
    )
    env1.eglobals[term.name] = ElabType(ty_term, implicit_spine, binder_names)
    body_term, body_ty = elab_infer(term.body, env1, state)
    return Let(ty_term, val_term, body_term), body_ty


def _elab_apply(
    fn: SurfaceTerm,
    args: tuple[SArg, ...],
    env: ElabEnv,
    state: ElabState,
    span: Span,
    *,
    allow_partial: bool,
) -> tuple[Term, ElabType]:
    fn_term, fn_ty = elab_infer(fn, env, state)
    implicit_spine = _implicit_spine_for_term(fn_term, env)
    binder_names = fn_ty.binder_names
    spine_index = 0
    context_env = env
    current_env = env
    missing_binders: list[tuple[Term, str | None]] = []
    named_seen = False
    for item in args:
        if item.name is not None:
            named_seen = True
            continue
        if named_seen:
            raise SurfaceError(
                "Positional arguments must come before named arguments",
                item.term.span,
            )
    if any(arg.name is not None for arg in args) and not binder_names:
        raise SurfaceError("Named arguments require binder names", span)
    positional = [arg for arg in args if arg.name is None]
    named: dict[str, SArg] = {}
    for item in args:
        if item.name is None:
            continue
        if item.name in named:
            raise SurfaceError(f"Duplicate named argument {item.name}", item.term.span)
        named[item.name] = item
    pos_index = 0
    while True:
        fn_ty_whnf = fn_ty.term.whnf(current_env.kenv)
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
                    current_env.kenv, fn_ty_whnf.arg_ty, span, kind="implicit"
                )
                fn_term = App(fn_term, meta)
                fn_ty = ElabType(
                    fn_ty_whnf.return_ty.subst(meta),
                    fn_ty.implicit_spine[1:],
                    fn_ty.binder_names[1:],
                )
                spine_index += 1
                continue
            if allow_partial and pos_index >= len(positional) and not named:
                break
            if allow_partial:
                missing_binders.append((fn_ty_whnf.arg_ty, binder_name))
                current_env = current_env.push_binder(
                    ElabType(fn_ty_whnf.arg_ty), name=binder_name
                )
                context_env = context_env.push_binder(
                    ElabType(fn_ty_whnf.arg_ty), name=binder_name
                )
                fn_term = fn_term.shift(1)
                fn_ty = ElabType(
                    fn_ty.term.shift(1), fn_ty.implicit_spine, fn_ty.binder_names
                )
                fn_ty_whnf = fn_ty.term.whnf(current_env.kenv)
                if not isinstance(fn_ty_whnf, Pi):
                    raise SurfaceError("Application of non-function", span)
                missing_term = Var(0)
                fn_term = App(fn_term, missing_term)
                fn_ty = ElabType(
                    fn_ty_whnf.return_ty.subst(missing_term),
                    fn_ty.implicit_spine[1:],
                    fn_ty.binder_names[1:],
                )
                spine_index += 1
                continue
            raise SurfaceError("Missing explicit argument", span)
        if not binder_is_implicit and arg.implicit and implicit_spine is not None:
            raise SurfaceError(
                "Implicit argument provided where explicit expected", arg.term.span
            )
        arg_term = elab_check(arg.term, context_env, state, ElabType(fn_ty_whnf.arg_ty))
        fn_term = App(fn_term, arg_term)
        fn_ty = ElabType(
            fn_ty_whnf.return_ty.subst(arg_term),
            fn_ty.implicit_spine[1:],
            fn_ty.binder_names[1:],
        )
        if binder_name is not None:
            context_env = context_env.with_context_term(
                binder_name,
                arg_term,
                ElabType(fn_ty_whnf.arg_ty),
            )
        spine_index += 1
        if consume_positional:
            pos_index += 1
    if named:
        unknown = next(iter(named.keys()))
        raise SurfaceError(f"Unknown named argument {unknown}", span)
    while True:
        fn_ty_whnf = fn_ty.term.whnf(current_env.kenv)
        if not isinstance(fn_ty_whnf, Pi):
            break
        if implicit_spine is None or spine_index >= len(implicit_spine):
            break
        if not implicit_spine[spine_index]:
            if allow_partial:
                break
            raise SurfaceError("Missing explicit argument", span)
        meta = state.fresh_meta(
            current_env.kenv, fn_ty_whnf.arg_ty, span, kind="implicit"
        )
        fn_term = App(fn_term, meta)
        fn_ty = ElabType(
            fn_ty_whnf.return_ty.subst(meta),
            fn_ty.implicit_spine[1:],
            fn_ty.binder_names[1:],
        )
        spine_index += 1
    if missing_binders:
        for ty, name in reversed(missing_binders):
            fn_term = Lam(ty, fn_term)
            fn_ty = ElabType(
                Pi(ty, fn_ty.term),
                (False,) + fn_ty.implicit_spine,
                (name,) + fn_ty.binder_names,
            )
    return fn_term, fn_ty


def _elab_binders(
    env: ElabEnv, state: ElabState, binders: tuple[SBinder, ...]
) -> tuple[list[Term], list[bool], list[LevelExpr], ElabEnv]:
    binder_tys: list[Term] = []
    binder_impls: list[bool] = []
    binder_levels: list[LevelExpr] = []
    for binder in binders:
        if binder.ty is None:
            raise SurfaceError("Missing binder type", binder.span)
        ty_term, ty_ty = elab_infer(binder.ty, env, state)
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
        names.extend(_normalize_binder_name(b.name) for b in current.binders)
        current = current.body
    return tuple(names)


def _binder_names_from_term(term: SurfaceTerm | None) -> tuple[str | None, ...]:
    if term is None:
        return ()
    match term:
        case SLam(binders=binders):
            return tuple(_normalize_binder_name(b.name) for b in binders)
        case SPi():
            return _binder_names(term)
    return ()


def _normalize_binder_name(name: str | None) -> str | None:
    if name == "_":
        return None
    return name


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
