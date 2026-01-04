"""Application elaboration helpers."""

from __future__ import annotations

from mltt.kernel.ast import App, Lam, Pi, Term, Var
from mltt.kernel.env import Env
from mltt.kernel.tel import ArgList
from typing import Iterable

from mltt.elab.east import (
    EAnn,
    EApp,
    EArg,
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
    ETerm,
    EUApp,
    EUniv,
    EVar,
)
from mltt.elab.elab_state import ElabState
from mltt.elab.etype import ElabBinderInfo, ElabEnv, ElabType
from mltt.surface.sast import Span, SurfaceError


def elab_apply(
    fn: ETerm,
    args: tuple[EArg, ...],
    env: ElabEnv,
    state: ElabState,
    span: Span,
    *,
    allow_partial: bool,
) -> tuple[Term, ElabType]:
    fn_term, fn_ty = _elab_infer(fn, env, state)
    remaining_binders = fn_ty.binders
    fn_term_closed = fn_term
    fn_ty_closed = fn_ty
    fn_ty_ctx = fn_ty.term
    ctx_env = env
    base_ctx_len = len(env.binders)
    actuals: list[Term] = []
    missing_binders: list[tuple[Term, str | None]] = []
    missing_depth = 0
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
    if any(arg.name is not None for arg in args) and not any(
        binder.name for binder in remaining_binders
    ):
        raise SurfaceError("Named arguments require binder names", span)
    positional = [arg for arg in args if arg.name is None]
    named: dict[str, EArg] = {}
    for item in args:
        if item.name is None:
            continue
        if item.name in named:
            raise SurfaceError(f"Duplicate named argument {item.name}", item.term.span)
        named[item.name] = item
    pos_index = 0
    while True:
        fn_ty_ctx_whnf = fn_ty_ctx.whnf(ctx_env.kenv)
        if not isinstance(fn_ty_ctx_whnf, Pi):
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
        if remaining_binders:
            binder_info = remaining_binders[0]
            remaining_binders = remaining_binders[1:]
        else:
            binder_info = ElabBinderInfo()
        binder_name = binder_info.name
        binder_is_implicit = binder_info.implicit
        arg: EArg | None = None
        if binder_name is not None and binder_name in named:
            arg = named.pop(binder_name)
        elif pos_index < len(positional):
            candidate = positional[pos_index]
            if binder_is_implicit:
                if candidate.implicit:
                    arg = candidate
                    pos_index += 1
            else:
                if candidate.implicit:
                    raise SurfaceError(
                        "Implicit argument provided where explicit expected",
                        candidate.term.span,
                    )
                arg = candidate
                pos_index += 1
        if arg is None:
            if binder_is_implicit:
                before_constraints = len(state.constraints)
                before_metas = set(state.metas.keys())
                meta_term_ctx = state.fresh_meta(
                    ctx_env.kenv, fn_ty_ctx_whnf.arg_ty, span, kind="implicit"
                )
                arg_term_closed = _close_term(meta_term_ctx, actuals)
                fn_term_closed = App(fn_term_closed, arg_term_closed)
                fn_ty_closed = _apply_fn_type(
                    fn_ty_closed, arg_term_closed, ctx_env.kenv, remaining_binders, span
                )
                fn_ty_ctx = fn_ty_ctx_whnf.return_ty.subst(meta_term_ctx)
                _close_new_constraints(
                    state, before_constraints, actuals, base_ctx_len, missing_depth
                )
                _close_new_metas(
                    state, before_metas, actuals, base_ctx_len, missing_depth
                )
                continue
            if allow_partial and pos_index >= len(positional) and not named:
                break
            if allow_partial:
                actuals = _shift_terms(actuals, 1)
                missing_ty = _close_term(fn_ty_ctx_whnf.arg_ty, actuals)
                missing_binders.append((missing_ty, binder_name))
                missing_depth += 1
                fn_term_closed = fn_term_closed.shift(1)
                fn_ty_closed = fn_ty_closed.shift(1)
                ctx_env = ctx_env.push_binder(
                    ElabType(fn_ty_ctx_whnf.arg_ty), name=binder_name
                )
                fn_ty_ctx = fn_ty_ctx_whnf.return_ty
                arg_term_closed = Var(0)
                fn_term_closed = App(fn_term_closed, arg_term_closed)
                fn_ty_closed = _apply_fn_type(
                    fn_ty_closed, arg_term_closed, ctx_env.kenv, remaining_binders, span
                )
                actuals.append(arg_term_closed)
                continue
            raise SurfaceError("Missing explicit argument", span)
        before_constraints = len(state.constraints)
        before_metas = set(state.metas.keys())
        arg_term_ctx = _elab_check(
            arg.term, ctx_env, state, ElabType(fn_ty_ctx_whnf.arg_ty)
        )
        _close_new_constraints(
            state, before_constraints, actuals, base_ctx_len, missing_depth
        )
        _close_new_metas(state, before_metas, actuals, base_ctx_len, missing_depth)
        arg_term_closed = _close_term(arg_term_ctx, actuals)
        fn_term_closed = App(fn_term_closed, arg_term_closed)
        fn_ty_closed = _apply_fn_type(
            fn_ty_closed, arg_term_closed, ctx_env.kenv, remaining_binders, span
        )
        binding_name = binder_name
        if binder_name is not None:
            if ctx_env.lookup_local(binder_name) is not None:
                binding_name = None
            if isinstance(arg_term_ctx, Var):
                existing = ctx_env.lookup_local(binder_name)
                if existing == arg_term_ctx.k:
                    binding_name = None
            if binding_name is not None and not _name_used_in_args(
                binder_name,
                positional[pos_index:],
                named.values(),
            ):
                binding_name = None
        ctx_env = ctx_env.push_let(
            ElabType(fn_ty_ctx_whnf.arg_ty),
            arg_term_ctx,
            name=binding_name,
        )
        fn_ty_ctx = fn_ty_ctx_whnf.return_ty
        actuals.append(arg_term_closed)
    if named:
        unknown = next(iter(named.keys()))
        raise SurfaceError(f"Unknown named argument {unknown}", span)
    if missing_binders:
        for ty, name in reversed(missing_binders):
            fn_term_closed = Lam(ty, fn_term_closed)
            fn_ty_closed = ElabType(
                Pi(ty, fn_ty_closed.term),
                (ElabBinderInfo(name=name, implicit=False),) + remaining_binders,
            )
            remaining_binders = fn_ty_closed.binders
    return fn_term_closed, fn_ty_closed


def _elab_infer(term: ETerm, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
    from mltt.elab.sast import elab_infer

    return elab_infer(term, env, state)


def _elab_check(
    term: ETerm, env: ElabEnv, state: ElabState, expected: ElabType
) -> Term:
    from mltt.elab.sast import elab_check

    return elab_check(term, env, state, expected)


def _close_term(term: Term, actuals: list[Term]) -> Term:
    if not actuals:
        return term
    return term.instantiate(ArgList.of(*actuals), depth_above=0)


def _shift_terms(terms: list[Term], amount: int) -> list[Term]:
    if amount == 0:
        return list(terms)
    return [term.shift(amount) for term in terms]


def _name_used_in_args(
    name: str,
    positional: list[EArg],
    named: Iterable[EArg],
) -> bool:
    for arg in positional:
        if _term_mentions_name(arg.term, name):
            return True
    for arg in named:
        if _term_mentions_name(arg.term, name):
            return True
    return False


def _term_mentions_name(term: ETerm, name: str) -> bool:
    match term:
        case EVar(name=var):
            return var == name
        case EConst() | EHole() | EUniv():
            return False
        case EAnn(term=inner, ty=ty):
            return _term_mentions_name(inner, name) or _term_mentions_name(ty, name)
        case EApp(fn=fn, args=args):
            if _term_mentions_name(fn, name):
                return True
            return any(_term_mentions_name(arg.term, name) for arg in args)
        case EUApp(head=head, levels=_):
            return _term_mentions_name(head, name)
        case EPartial(term=inner):
            return _term_mentions_name(inner, name)
        case ELam(binders=binders, body=body):
            return any(
                (b.ty is not None and _term_mentions_name(b.ty, name))
                or (b.name == name)
                for b in binders
            ) or _term_mentions_name(body, name)
        case EPi(binders=binders, body=body):
            return any(
                (b.ty is not None and _term_mentions_name(b.ty, name))
                or (b.name == name)
                for b in binders
            ) or _term_mentions_name(body, name)
        case ELet(name=_, uparams=_, params=params, ty=ty, val=val, body=body):
            if ty is not None and _term_mentions_name(ty, name):
                return True
            if _term_mentions_name(val, name):
                return True
            if any(
                (b.ty is not None and _term_mentions_name(b.ty, name))
                or (b.name == name)
                for b in params
            ):
                return True
            return _term_mentions_name(body, name)
        case EMatch(scrutinee=scrutinee, branches=branches, motive=motive):
            if _term_mentions_name(scrutinee, name):
                return True
            if motive is not None and _term_mentions_name(motive, name):
                return True
            return any(_term_mentions_name(branch.rhs, name) for branch in branches)
        case EInductiveDef(
            name=_,
            uparams=_,
            params=params,
            level=level,
            ctors=ctors,
            body=body,
        ):
            if any(
                (b.ty is not None and _term_mentions_name(b.ty, name))
                or (b.name == name)
                for b in params
            ):
                return True
            if _term_mentions_name(level, name):
                return True
            if any(
                _term_mentions_name(ctor.result, name)
                or any(
                    b.ty is not None and _term_mentions_name(b.ty, name)
                    for b in ctor.fields
                )
                for ctor in ctors
            ):
                return True
            return _term_mentions_name(body, name)
        case EInd() | ECtor():
            return False
    return False


def _close_new_constraints(
    state: ElabState,
    start: int,
    actuals: list[Term],
    base_ctx_len: int,
    missing_depth: int,
) -> None:
    if start >= len(state.constraints):
        return
    actuals_arglist = ArgList.of(*actuals) if actuals else None
    for constraint in state.constraints[start:]:
        if actuals_arglist is not None:
            constraint.lhs = constraint.lhs.instantiate(actuals_arglist, depth_above=0)
            constraint.rhs = constraint.rhs.instantiate(actuals_arglist, depth_above=0)
        constraint.ctx_len = base_ctx_len + missing_depth


def _close_new_metas(
    state: ElabState,
    before: set[int],
    actuals: list[Term],
    base_ctx_len: int,
    missing_depth: int,
) -> None:
    new_meta_ids = set(state.metas.keys()) - before
    if not new_meta_ids:
        return
    actuals_arglist = ArgList.of(*actuals) if actuals else None
    for mid in new_meta_ids:
        meta = state.metas[mid]
        if actuals_arglist is not None:
            meta.ty = meta.ty.instantiate(actuals_arglist, depth_above=0)
            if meta.solution is not None:
                meta.solution = meta.solution.instantiate(
                    actuals_arglist, depth_above=0
                )
        meta.ctx_len = base_ctx_len + missing_depth


def _apply_fn_type(
    fn_ty: ElabType,
    arg_term: Term,
    env: Env,
    remaining_binders: tuple[ElabBinderInfo, ...],
    span: Span,
) -> ElabType:
    fn_ty_whnf = fn_ty.term.whnf(env)
    if not isinstance(fn_ty_whnf, Pi):
        raise SurfaceError("Application of non-function", span)
    return ElabType(fn_ty_whnf.return_ty.subst(arg_term), remaining_binders)
