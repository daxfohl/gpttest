"""Milestone 1 elaboration: explicit binders + check-mode lambdas."""

from __future__ import annotations

from mltt.kernel.ast import App, Lam, Let, Pi, Term, Univ, Var
from mltt.kernel.environment import Const, Env
from mltt.kernel.telescope import mk_app, mk_pis
from mltt.surface.syntax import (
    SurfaceError,
    SurfaceTerm,
    SBinder,
    SVar,
    SType,
    SAnn,
    SLam,
    SPi,
    SApp,
    SLet,
)


def elab_infer(env: Env, st: SurfaceTerm) -> tuple[Term, Term]:
    if isinstance(st, SVar):
        idx = env.lookup_local(st.name)
        if idx is not None:
            return Var(idx), env.local_type(idx)
        decl = env.lookup_global(st.name)
        if decl is not None:
            return Const(st.name), decl.ty
        raise SurfaceError(f"Unknown identifier {st.name}", st.span)
    if isinstance(st, SType):
        term_out = Univ(st.level)
        return term_out, Univ(st.level + 1)
    if isinstance(st, SAnn):
        ty_term, _ = elab_infer(env, st.ty)
        _ = ty_term.expect_universe(env)
        checked_term = elab_check(env, st.term, ty_term)
        return checked_term, ty_term
    if isinstance(st, SApp):
        fn_term, fn_ty = elab_infer(env, st.fn)
        for arg in st.args:
            fn_ty_whnf = fn_ty.whnf(env)
            if not isinstance(fn_ty_whnf, Pi):
                raise SurfaceError("Application of non-function", arg.span)
            arg_term = elab_check(env, arg, fn_ty_whnf.arg_ty)
            fn_term = App(fn_term, arg_term)
            fn_ty = fn_ty_whnf.return_ty.subst(arg_term)
        return fn_term, fn_ty
    if isinstance(st, SPi):
        binder_tys, env1 = _elab_binders(env, st.binders)
        body_term, _ = elab_infer(env1, st.body)
        pi_term = mk_pis(*binder_tys, return_ty=body_term)
        return pi_term, pi_term.infer_type(env)
    if isinstance(st, SLam):
        if any(b.ty is None for b in st.binders):
            raise SurfaceError(
                "Cannot infer unannotated lambda; add binder types or use check-mode",
                st.span,
            )
        binder_tys, env1 = _elab_binders(env, st.binders)
        body_term, body_ty = elab_infer(env1, st.body)
        lam_term = body_term
        lam_ty = body_ty
        for ty in reversed(binder_tys):
            lam_term = Lam(ty, lam_term)
            lam_ty = Pi(ty, lam_ty)
        return lam_term, lam_ty
    if isinstance(st, SLet):
        ty_term, _ = elab_infer(env, st.ty)
        _ = ty_term.expect_universe(env)
        val_term = elab_check(env, st.val, ty_term)
        env1 = env.push_let(ty_term, val_term, name=st.name)
        body_term, body_ty = elab_infer(env1, st.body)
        return Let(ty_term, val_term, body_term), body_ty
    raise SurfaceError("Unsupported surface term", st.span)


def elab_check(env: Env, st: SurfaceTerm, expected: Term) -> Term:
    if isinstance(st, SLam):
        binder_tys: list[Term] = []
        env1 = env
        expected_ty = expected
        for binder in st.binders:
            pi_ty = expected_ty.whnf(env1)
            if not isinstance(pi_ty, Pi):
                raise SurfaceError("Lambda needs expected function type", st.span)
            if binder.ty is None:
                binder_ty = pi_ty.arg_ty
            else:
                binder_ty, _ = elab_infer(env1, binder.ty)
                _ = binder_ty.expect_universe(env1)
                if not binder_ty.type_equal(pi_ty.arg_ty, env1):
                    raise SurfaceError("Lambda binder type mismatch", binder.span)
            binder_tys.append(binder_ty)
            env1 = env1.push_binder(binder_ty, name=binder.name)
            expected_ty = pi_ty.return_ty
        body_term = elab_check(env1, st.body, expected_ty)
        lam_term = body_term
        for ty in reversed(binder_tys):
            lam_term = Lam(ty, lam_term)
        return lam_term
    term, term_ty = elab_infer(env, st)
    if not term_ty.type_equal(expected, env):
        raise SurfaceError("Type mismatch", st.span)
    return term


def _elab_binders(env: Env, binders: tuple[SBinder, ...]) -> tuple[list[Term], Env]:
    binder_tys: list[Term] = []
    env1 = env
    for binder in binders:
        if binder.ty is None:
            raise SurfaceError("Missing binder type", binder.span)
        ty_term, _ = elab_infer(env1, binder.ty)
        _ = ty_term.expect_universe(env1)
        binder_tys.append(ty_term)
        env1 = env1.push_binder(ty_term, name=binder.name)
    return binder_tys, env1
