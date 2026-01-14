"""Surface elaboration helpers."""

from __future__ import annotations

from mltt.common.span import Span
from mltt.elab.apply import elab_apply
from mltt.elab.ast import (
    EAnn,
    EApp,
    EBinder,
    EConst,
    ECtor,
    EHole,
    EInd,
    EInductiveDef,
    ELet,
    ELam,
    EMatch,
    EPartial,
    EPi,
    ELevel,
    EUniv,
    EUApp,
    EVar,
    ETerm,
)
from mltt.elab.errors import ElabError
from mltt.elab.generalize import generalize_let
from mltt.elab.names import NameEnv
from mltt.solver.solver import Solver
from mltt.elab.types import (
    BinderSpec,
    ElabEnv,
    ElabType,
    attach_binder_types,
    normalize_binder_name,
)
from mltt.kernel.ast import App, Lam, Let, Pi, Term, Univ, Var, UApp
from mltt.kernel.env import Const, Env, GlobalDecl
from mltt.kernel.ind import Ctor, Ind
from mltt.kernel.levels import LConst, LVar, LevelExpr


def elab_infer(term: ETerm, env: ElabEnv, solver: Solver) -> tuple[Term, ElabType]:
    match term:
        case EVar(name=name):
            idx = env.lookup_local(name)
            if idx is not None:
                binder = env.binders[idx]
                term_k, levels = solver.apply_implicit_levels(
                    Var(idx), binder.uarity, term.span
                )
                ty = env.local_type(idx).inst_levels(levels)
                return term_k, ElabType(solver.zonk(ty.term), ty.binders)
            decl, gty = require_global_info(
                env, name, term.span, f"Unknown identifier {name}"
            )
            head: Term
            if isinstance(decl.value, (Ind, Ctor)):
                head = decl.value
            else:
                head = Const(name)
            term_k, levels = solver.apply_implicit_levels(head, decl.uarity, term.span)
            ty = gty.inst_levels(levels)
            return term_k, ElabType(solver.zonk(ty.term), ty.binders)
        case EConst(name=name):
            decl, gty = require_global_info(
                env, name, term.span, f"Unknown constant {name}"
            )
            term_k, levels = solver.apply_implicit_levels(
                Const(name), decl.uarity, term.span
            )
            ty = gty.inst_levels(levels)
            return term_k, ElabType(solver.zonk(ty.term), ty.binders)
        case EUniv(level=level):
            level_expr: LevelExpr | int
            if level is None:
                level_expr = solver.fresh_level_meta("type", term.span)
            else:
                level_expr = _level_expr(level)
            term_k = Univ(level_expr)
            return term_k, ElabType(Univ(LevelExpr.of(level_expr).succ()))
        case EAnn(term=inner, ty=ty_src):
            ty_term, ty_ty = elab_infer(ty_src, env, solver)
            expect_universe(ty_ty.term, env.kenv, term.span)
            term_k = elab_check(inner, env, solver, ElabType(ty_term))
            return term_k, ElabType(ty_term)
        case EHole():
            raise ElabError("Hole needs expected type", term.span)
        case ELam():
            return _elab_lam_infer(term, env, solver)
        case EPi():
            return _elab_pi_infer(term, env, solver)
        case EApp(fn=fn, args=args, named_args=named_args):
            return elab_apply(
                fn, args, named_args, env, solver, term.span, allow_partial=False
            )
        case EUApp():
            return _elab_uapp_infer(term, env, solver)
        case EPartial(term=inner):
            return _elab_partial_infer(inner, env, solver, term.span)
        case ELet():
            return _elab_let_infer(term, env, solver)
        case EMatch():
            from mltt.elab.match import elab_match_infer

            return elab_match_infer(term, env, solver)
        case EInd():
            from mltt.elab.inductive import elab_ind_infer

            return elab_ind_infer(term, env, solver)
        case ECtor():
            from mltt.elab.inductive import elab_ctor_infer

            return elab_ctor_infer(term, env, solver)
        case EInductiveDef():
            from mltt.elab.inductive import elab_inductive_infer

            return elab_inductive_infer(term, env, solver)
        case _:
            raise ElabError("Unsupported surface term", term.span)


def elab_check(term: ETerm, env: ElabEnv, solver: Solver, expected: ElabType) -> Term:
    match term:
        case EHole():
            return solver.fresh_meta(env.kenv, expected.term, term.span, kind="hole")
        case ELam():
            return _elab_lam_check(term, env, solver, expected)
        case EMatch():
            from mltt.elab.match import elab_match_check

            return elab_match_check(term, env, solver, expected)
        case EAnn(term=inner, ty=ty_src):
            ty_term, ty_ty = elab_infer(ty_src, env, solver)
            expect_universe(ty_ty.term, env.kenv, term.span)
            term_k = elab_check(inner, env, solver, ElabType(ty_term))
            solver.add_constraint(env.kenv, ty_term, expected.term, term.span)
            return term_k
        case _:
            term_k, term_ty = elab_infer(term, env, solver)
            solver.add_constraint(env.kenv, term_ty.term, expected.term, term.span)
            return term_k


def _resolve(term: ETerm, env: Env, names: NameEnv) -> Term:
    match term:
        case EVar(name=name):
            idx = names.lookup(name)
            if idx is not None:
                return Var(idx)
            decl = env.lookup_global(name)
            if decl is not None:
                if isinstance(decl.value, (Ind, Ctor)):
                    return decl.value
                return Const(name)
            raise ElabError(f"Unknown identifier {name}", term.span)
        case EConst(name=name):
            if env.lookup_global(name) is None:
                raise ElabError(f"Unknown constant {name}", term.span)
            return Const(name)
        case EUniv(level=level):
            if level is None:
                raise ElabError("Universe requires elaboration", term.span)
            return Univ(_level_expr(level))
        case EAnn(term=inner, ty=ty_src):
            _ = ty_src
            return _resolve(inner, env, names)
        case EHole():
            raise ElabError("Hole requires elaboration", term.span)
        case ELam(binders=binders, body=body):
            if any(b.ty is None for b in binders):
                raise ElabError("Missing binder type", term.span)
            tys: list[Term] = []
            for binder in binders:
                assert binder.ty is not None
                tys.append(_resolve(binder.ty, env, names))
                names.push(binder.name)
            term_k = _resolve(body, env, names)
            for ty in reversed(tys):
                term_k = Lam(ty, term_k)
                names.pop()
            return term_k
        case EPi(binders=binders, body=body):
            if any(b.ty is None for b in binders):
                raise ElabError("Missing binder type", term.span)
            pi_tys: list[Term] = []
            for binder in binders:
                assert binder.ty is not None
                pi_tys.append(_resolve(binder.ty, env, names))
                names.push(binder.name)
            term_k = _resolve(body, env, names)
            for ty in reversed(pi_tys):
                term_k = Pi(ty, term_k)
                names.pop()
            return term_k
        case EApp(fn=fn, args=args, named_args=named_args):
            term_k = _resolve(fn, env, names)
            for arg in args:
                arg_term = _resolve(arg.term, env, names)
                term_k = App(term_k, arg_term)
            if named_args:
                raise ElabError("Named arguments require elaboration", term.span)
            return term_k
        case EUApp(head=head, levels=levels):
            head_term = _resolve(head, env, names)
            level_terms = tuple(_level_expr(level) for level in levels)
            return UApp(head_term, level_terms)
        case EPartial(term=inner):
            return _resolve(inner, env, names)
        case ELet(name=name, ty=ty_src, val=val_src, body=body):
            if ty_src is None:
                raise ElabError("Missing let type; needs elaboration", term.span)
            ty_term = _resolve(ty_src, env, names)
            val_term = _resolve(val_src, env, names)
            names.push(name)
            body_term = _resolve(body, env, names)
            names.pop()
            return Let(ty_term, val_term, body_term)
        case EMatch() | EInd() | ECtor() | EInductiveDef():
            raise ElabError("Surface construct requires elaboration", term.span)
        case _:
            raise ElabError("Unsupported surface term", term.span)


def expect_universe(term: Term, env: Env, span: Span) -> Univ:
    ty_whnf = term.whnf(env)
    if not isinstance(ty_whnf, Univ):
        raise ElabError(f"{type(term).__name__} must be a universe", span)
    return ty_whnf


def require_global_info(
    env: ElabEnv, name: str, span: Span, message: str
) -> tuple[GlobalDecl, ElabType]:
    info = env.global_info(name)
    if info is None:
        raise ElabError(message, span)
    return info


def _elab_lam_infer(term: ELam, env: ElabEnv, solver: Solver) -> tuple[Term, ElabType]:
    if any(b.ty is None for b in term.binders):
        raise ElabError(
            "Cannot infer unannotated lambda; add binder types or use check-mode",
            term.span,
        )
    binder_tys, binder_impls, _binder_levels, env1 = elab_binders(
        env, solver, term.binders
    )
    body_term, body_ty = elab_infer(term.body, env1, solver)
    lam_term = body_term
    lam_ty_term = body_ty.term
    binder_infos = body_ty.binders
    for ty, implicit, name in reversed(
        list(zip(binder_tys, binder_impls, (b.name for b in term.binders)))
    ):
        lam_term = Lam(ty, lam_term)
        lam_ty_term = Pi(ty, lam_ty_term)
        binder_infos = (
            BinderSpec(normalize_binder_name(name), implicit, ty),
        ) + binder_infos
    return lam_term, ElabType(lam_ty_term, binder_infos)


def _elab_lam_check(
    term: ELam, env: ElabEnv, solver: Solver, expected: ElabType
) -> Term:
    binder_tys: list[Term] = []
    binder_impls: list[bool] = []
    env1 = env
    expected_ty = expected
    for binder in term.binders:
        pi_ty = expected_ty.term.whnf(env1.kenv)
        if not isinstance(pi_ty, Pi):
            raise ElabError("Lambda needs expected function type", term.span)
        binder_ty_info: tuple[BinderSpec, ...] = ()
        if binder.ty is None:
            binder_ty = pi_ty.arg_ty
        else:
            binder_ty, binder_ty_ty = elab_infer(binder.ty, env1, solver)
            expect_universe(binder_ty_ty.term, env1.kenv, binder.span)
            solver.add_constraint(env1.kenv, binder_ty, pi_ty.arg_ty, binder.span)
            binder_ty_info = attach_binder_types(
                binder_ty, _binder_specs_from_type(binder.ty), env1.kenv
            )
        binder_tys.append(binder_ty)
        binder_impls.append(binder.implicit)
        env1 = env1.push_binder(ElabType(binder_ty, binder_ty_info), name=binder.name)
        expected_ty = ElabType(pi_ty.return_ty, expected_ty.binders[1:])
    body_term = elab_check(term.body, env1, solver, expected_ty)
    lam_term = body_term
    for ty in reversed(binder_tys):
        lam_term = Lam(ty, lam_term)
    return lam_term


def _elab_pi_infer(term: EPi, env: ElabEnv, solver: Solver) -> tuple[Term, ElabType]:
    binder_tys, binder_impls, binder_levels, env1 = elab_binders(
        env, solver, term.binders
    )
    body_term, body_ty = elab_infer(term.body, env1, solver)
    body_ty_whnf = expect_universe(body_ty.term, env1.kenv, term.body.span)
    body_level = body_ty_whnf.level
    pi_term: Term = body_term
    result_level: LevelExpr | None = None
    for ty, arg_level in reversed(list(zip(binder_tys, binder_levels))):
        if result_level is None:
            result_level = solver.fresh_level_meta("type", term.span)
        solver.add_level_constraint(arg_level, result_level, term.span)
        solver.add_level_constraint(body_level, result_level, term.span)
        body_level = result_level
        pi_term = Pi(ty, pi_term)
    if result_level is None:
        result_level = solver.fresh_level_meta("type", term.span)
        solver.add_level_constraint(body_level, result_level, term.span)
    binder_infos = tuple(
        BinderSpec(normalize_binder_name(b.name), b.implicit, ty)
        for b, ty in zip(term.binders, binder_tys, strict=True)
    )
    return pi_term, ElabType(Univ(result_level), binder_infos)


def _elab_uapp_infer(
    term: EUApp, env: ElabEnv, solver: Solver
) -> tuple[Term, ElabType]:
    head_term: Term
    match term.head:
        case EVar(name=name):
            if env.lookup_local(name) is not None:
                raise ElabError("UApp head must be a global", term.span)
            decl = env.lookup_global(name)
            if decl is None:
                raise ElabError(f"Unknown identifier {name}", term.span)
            if isinstance(decl.value, (Ind, Ctor)):
                head_term = decl.value
            else:
                head_term = Const(name)
        case EConst(name=name):
            decl = env.lookup_global(name)
            if decl is None:
                raise ElabError(f"Unknown constant {name}", term.span)
            head_term = Const(name)
        case EInd(name=name):
            decl = env.lookup_global(name)
            if decl is None or decl.value is None:
                raise ElabError(f"Unknown inductive {name}", term.span)
            head_term = decl.value
            if isinstance(head_term, UApp) and isinstance(head_term.head, Ind):
                head_term = head_term.head
        case ECtor(name=name):
            decl = env.lookup_global(name)
            if decl is None or decl.value is None:
                raise ElabError(f"Unknown constructor {name}", term.span)
            head_term = decl.value
            if isinstance(head_term, UApp) and isinstance(head_term.head, Ctor):
                head_term = head_term.head
        case _:
            head_term, _ = elab_infer(term.head, env, solver)
            if isinstance(head_term, UApp):
                head_term = head_term.head
    level_terms: tuple[LevelExpr, ...] = tuple(
        _level_expr(level) for level in term.levels
    )
    uapp = UApp(head_term, level_terms)
    return uapp, ElabType(uapp.infer_type(env.kenv))


def _level_expr(level: ELevel) -> LevelExpr:
    match level:
        case ELevel(kind="const", value=value):
            return LConst(value)
        case ELevel(kind="bound", value=value):
            return LVar(value)
    raise ValueError("Unknown level expression")


def _elab_partial_infer(
    term: ETerm, env: ElabEnv, solver: Solver, span: Span
) -> tuple[Term, ElabType]:
    match term:
        case EApp(fn=fn, args=args, named_args=named_args):
            return elab_apply(
                fn, args, named_args, env, solver, span, allow_partial=True
            )
    return elab_infer(term, env, solver)


def _elab_let_infer(term: ELet, env: ElabEnv, solver: Solver) -> tuple[Term, ElabType]:
    if len(set(term.uparams)) != len(term.uparams):
        raise ElabError("Duplicate universe binder", term.span)
    val_src = term.val
    if term.ty is None:
        val_term, val_ty = elab_infer(val_src, env, solver)
        ty_term = val_ty.term
        binder_infos = val_ty.binders
    else:
        ty_term, ty_ty = elab_infer(term.ty, env, solver)
        expect_universe(ty_ty.term, env.kenv, term.span)
        binder_infos = attach_binder_types(
            ty_term, _binder_specs_from_type(term.ty), env.kenv
        )
        val_term = elab_check(val_src, env, solver, ElabType(ty_term))
    uarity, ty_term, val_term = generalize_let(solver, ty_term, val_term)
    env1 = env.push_let(
        ElabType(ty_term, binder_infos),
        val_term,
        name=term.name,
        uarity=uarity,
    )
    env1.eglobals[term.name] = ElabType(ty_term, binder_infos)
    body_term, body_ty = elab_infer(term.body, env1, solver)
    return Let(ty_term, val_term, body_term), body_ty


def elab_binders(
    env: ElabEnv, solver: Solver, binders: tuple[EBinder, ...]
) -> tuple[list[Term], list[bool], list[LevelExpr], ElabEnv]:
    binder_tys: list[Term] = []
    binder_impls: list[bool] = []
    binder_levels: list[LevelExpr] = []
    for binder in binders:
        if binder.ty is None:
            raise ElabError("Missing binder type", binder.span)
        ty_term, ty_ty = elab_infer(binder.ty, env, solver)
        ty_ty_whnf = expect_universe(ty_ty.term, env.kenv, binder.span)
        binder_infos = attach_binder_types(
            ty_term, _binder_specs_from_type(binder.ty), env.kenv
        )
        env = env.push_binder(ElabType(ty_term, binder_infos), name=binder.name)
        binder_tys.append(ty_term)
        binder_impls.append(binder.implicit)
        binder_levels.append(ty_ty_whnf.level)
    return binder_tys, binder_impls, binder_levels, env


def _binder_specs_from_type(term: ETerm | None) -> tuple[BinderSpec, ...]:
    if term is None:
        return ()
    infos: list[BinderSpec] = []
    current = term
    while isinstance(current, EPi):
        infos.extend(
            BinderSpec(
                name=normalize_binder_name(b.name),
                implicit=b.implicit,
            )
            for b in current.binders
        )
        current = current.body
    return tuple(infos)
