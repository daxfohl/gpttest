"""Application elaboration helpers."""

from __future__ import annotations

from typing import Iterable

from mltt.common.span import Span
from mltt.elab.apply_matcher import ArgMatcher
from mltt.elab.ast import (
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
    ENamedArg,
    EPartial,
    EPi,
    ETerm,
    EUApp,
    EUniv,
    EVar,
)
from mltt.elab.errors import ElabError
from mltt.elab.state import ElabState
from mltt.elab.types import BinderSpec, ElabEnv, ElabType, apply_binder_specs
from mltt.kernel.ast import App, Lam, Let, MetaVar, Pi, Term, UApp, Univ, Var
from mltt.kernel.env import Const, Env
from mltt.kernel.ind import Ctor, Elim, Ind
from mltt.kernel.tel import Spine


def elab_apply(
    fn: ETerm,
    args: tuple[EArg, ...],
    named_args: tuple[ENamedArg, ...],
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
    matcher = ArgMatcher(remaining_binders, args, named_args, span)
    while True:
        fn_ty_ctx_whnf = fn_ty_ctx.whnf(ctx_env.kenv)
        if not isinstance(fn_ty_ctx_whnf, Pi):
            if matcher.has_positional() or matcher.has_named():
                raise ElabError(
                    "Application of non-function",
                    matcher.next_arg_span(),
                )
            break
        if remaining_binders:
            binder_info = remaining_binders[0]
            remaining_binders = remaining_binders[1:]
        else:
            binder_info = BinderSpec()
        binder_info = BinderSpec(
            name=binder_info.name,
            implicit=binder_info.implicit,
            ty=fn_ty_ctx_whnf.arg_ty,
        )
        decision = matcher.match_for_binder(binder_info, allow_partial=allow_partial)
        match decision.kind:
            case "stop":
                break
            case "implicit":
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
            case "missing":
                if not allow_partial:
                    missing_name = binder_info.name or "<unnamed>"
                    dependent = _dependent_binder_names(remaining_binders)
                    if dependent:
                        suffix = ", ".join(dependent)
                        message = (
                            f"Missing explicit argument {missing_name}; "
                            f"later arguments depend on it: {suffix}"
                        )
                    else:
                        message = f"Missing explicit argument {missing_name}"
                    raise ElabError(message, matcher.next_arg_span())
                actuals = _shift_terms(actuals, 1)
                missing_ty = _close_term(fn_ty_ctx_whnf.arg_ty, actuals)
                missing_binders.append((missing_ty, binder_info.name))
                missing_depth += 1
                fn_term_closed = fn_term_closed.shift(1)
                fn_ty_closed = fn_ty_closed.shift(1)
                ctx_env = ctx_env.push_binder(
                    ElabType(fn_ty_ctx_whnf.arg_ty), name=binder_info.name
                )
                fn_ty_ctx = fn_ty_ctx_whnf.return_ty
                arg_term_closed = Var(0)
                fn_term_closed = App(fn_term_closed, arg_term_closed)
                fn_ty_closed = _apply_fn_type(
                    fn_ty_closed, arg_term_closed, ctx_env.kenv, remaining_binders, span
                )
                actuals.append(arg_term_closed)
                continue
            case "explicit":
                assert isinstance(decision.arg, (EArg, ENamedArg))
                if not binder_info.implicit and isinstance(decision.arg, EArg):
                    if decision.arg.implicit:
                        raise ElabError(
                            "Implicit argument provided where explicit expected",
                            decision.arg.term.span,
                        )
                before_constraints = len(state.constraints)
                before_metas = set(state.metas.keys())
                arg_term_ctx = _elab_check(
                    decision.arg.term,
                    ctx_env,
                    state,
                    ElabType(fn_ty_ctx_whnf.arg_ty),
                )
                _close_new_constraints(
                    state, before_constraints, actuals, base_ctx_len, missing_depth
                )
                _close_new_metas(
                    state, before_metas, actuals, base_ctx_len, missing_depth
                )
                arg_term_closed = _close_term(arg_term_ctx, actuals)
                fn_term_closed = App(fn_term_closed, arg_term_closed)
                fn_ty_closed = _apply_fn_type(
                    fn_ty_closed, arg_term_closed, ctx_env.kenv, remaining_binders, span
                )
                binding_name = binder_info.name
                if binding_name is not None:
                    if ctx_env.lookup_local(binding_name) is not None:
                        binding_name = None
                if binding_name is not None and isinstance(arg_term_ctx, Var):
                    existing = ctx_env.lookup_local(binding_name)
                    if existing == arg_term_ctx.k:
                        binding_name = None
                if binding_name is not None and not _name_used_in_args(
                    binding_name,
                    matcher.remaining_positional(),
                    matcher.remaining_named(),
                ):
                    binding_name = None
                ctx_env = ctx_env.push_let(
                    ElabType(fn_ty_ctx_whnf.arg_ty),
                    arg_term_ctx,
                    name=binding_name,
                )
                fn_ty_ctx = fn_ty_ctx_whnf.return_ty
                actuals.append(arg_term_closed)
                continue
    if matcher.has_named():
        unknown = matcher.unknown_named()
        assert unknown is not None
        raise ElabError(f"Unknown named argument {unknown}", span)
    if missing_binders:
        for ty, name in reversed(missing_binders):
            fn_term_closed = Lam(ty, fn_term_closed)
            fn_ty_closed = ElabType(
                Pi(ty, fn_ty_closed.term),
                (BinderSpec(name=name, implicit=False, ty=ty),) + remaining_binders,
            )
            remaining_binders = fn_ty_closed.binders
    return fn_term_closed, fn_ty_closed


def _elab_infer(term: ETerm, env: ElabEnv, state: ElabState) -> tuple[Term, ElabType]:
    from mltt.elab.term import elab_infer

    return elab_infer(term, env, state)


def _elab_check(
    term: ETerm, env: ElabEnv, state: ElabState, expected: ElabType
) -> Term:
    from mltt.elab.term import elab_check

    return elab_check(term, env, state, expected)


def _close_term(term: Term, actuals: list[Term]) -> Term:
    if not actuals:
        return term
    return term.instantiate(Spine.of(*actuals), depth_above=0)


def _shift_terms(terms: list[Term], amount: int) -> list[Term]:
    if amount == 0:
        return list(terms)
    return [term.shift(amount) for term in terms]


def _dependent_binder_names(
    binders: tuple[BinderSpec, ...],
) -> list[str]:
    names: list[str] = []
    for idx, spec in enumerate(binders):
        if spec.ty is not None and _term_uses_var(spec.ty, idx):
            names.append(spec.name or "<unnamed>")
    return names


def _term_uses_var(term: Term, target: int, depth: int = 0) -> bool:
    match term:
        case Var(k):
            return k == target + depth
        case Lam(ty, body) | Pi(ty, body):
            return _term_uses_var(ty, target, depth) or _term_uses_var(
                body, target, depth + 1
            )
        case Let(arg_ty, value, body):
            return (
                _term_uses_var(arg_ty, target, depth)
                or _term_uses_var(value, target, depth)
                or _term_uses_var(body, target, depth + 1)
            )
        case App(f, a):
            return _term_uses_var(f, target, depth) or _term_uses_var(a, target, depth)
        case UApp(head, _levels):
            return _term_uses_var(head, target, depth)
        case Elim(inductive, motive, cases, scrutinee):
            return (
                _term_uses_var(inductive, target, depth)
                or _term_uses_var(motive, target, depth)
                or any(_term_uses_var(case, target, depth) for case in cases)
                or _term_uses_var(scrutinee, target, depth)
            )
        case Const() | Ind() | Ctor() | MetaVar() | Univ():
            return False
        case _:
            return False


def _name_used_in_args(
    name: str,
    positional: list[EArg],
    named: Iterable[ENamedArg],
) -> bool:
    for pos_arg in positional:
        if _term_mentions_name(pos_arg.term, name):
            return True
    for named_arg in named:
        if _term_mentions_name(named_arg.term, name):
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
        case EApp(fn=fn, args=args, named_args=named_args):
            if _term_mentions_name(fn, name):
                return True
            if any(_term_mentions_name(arg.term, name) for arg in args):
                return True
            return any(_term_mentions_name(arg.term, name) for arg in named_args)
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
        case ELet(name=_, uparams=_, ty=ty, val=val, body=body):
            if ty is not None and _term_mentions_name(ty, name):
                return True
            if _term_mentions_name(val, name):
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
    actuals_spine = Spine.of(*actuals) if actuals else None
    for constraint in state.constraints[start:]:
        if actuals_spine is not None:
            constraint.lhs = constraint.lhs.instantiate(actuals_spine, depth_above=0)
            constraint.rhs = constraint.rhs.instantiate(actuals_spine, depth_above=0)
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
    actuals_spine = Spine.of(*actuals) if actuals else None
    for mid in new_meta_ids:
        meta = state.metas[mid]
        if actuals_spine is not None:
            meta.ty = meta.ty.instantiate(actuals_spine, depth_above=0)
            if meta.solution is not None:
                meta.solution = meta.solution.instantiate(actuals_spine, depth_above=0)
        meta.ctx_len = base_ctx_len + missing_depth


def _apply_fn_type(
    fn_ty: ElabType,
    arg_term: Term,
    env: Env,
    remaining_binders: tuple[BinderSpec, ...],
    span: Span,
) -> ElabType:
    fn_ty_whnf = fn_ty.term.whnf(env)
    if not isinstance(fn_ty_whnf, Pi):
        raise ElabError("Application of non-function", span)
    updated_binders = apply_binder_specs(remaining_binders, arg_term)
    return ElabType(fn_ty_whnf.return_ty.subst(arg_term), updated_binders)
