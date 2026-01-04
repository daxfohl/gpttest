"""Application elaboration helpers."""

from __future__ import annotations

from mltt.kernel.ast import App, Lam, Pi, Term, UApp, Var
from mltt.kernel.env import Const
from mltt.kernel.ind import Ctor, Ind
from mltt.elab.elab_state import ElabState
from mltt.elab.etype import ElabEnv, ElabType
from mltt.surface.sast import SArg, Span, SurfaceError, SurfaceTerm


def elab_apply(
    fn: SurfaceTerm,
    args: tuple[SArg, ...],
    env: ElabEnv,
    state: ElabState,
    span: Span,
    *,
    allow_partial: bool,
) -> tuple[Term, ElabType]:
    fn_term, fn_ty = _elab_infer(fn, env, state)
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
        arg_term = _elab_check(
            arg.term, context_env, state, ElabType(fn_ty_whnf.arg_ty)
        )
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


def _elab_infer(
    term: SurfaceTerm, env: ElabEnv, state: ElabState
) -> tuple[Term, ElabType]:
    from mltt.elab.sast import elab_infer

    return elab_infer(term, env, state)


def _elab_check(
    term: SurfaceTerm, env: ElabEnv, state: ElabState, expected: ElabType
) -> Term:
    from mltt.elab.sast import elab_check

    return elab_check(term, env, state, expected)


def _implicit_spine_for_term(term: Term, env: ElabEnv) -> tuple[bool, ...] | None:
    head = term
    applied = 0
    while isinstance(head, App):
        applied += 1
        head = head.func
    if isinstance(head, UApp):
        head = head.head
    match head:
        case Var():
            implicit_spine = env.locals[head.k].implicit_spine
        case Const():
            gty = env.global_type(head.name)
            assert gty is not None
            implicit_spine = gty.implicit_spine
        case Ind():
            gty = env.global_type(head.name)
            if gty is None:
                return None
            implicit_spine = gty.implicit_spine
        case Ctor():
            ctor_name = f"{head.inductive.name}.{head.name}"
            gty = env.global_type(ctor_name)
            if gty is None:
                return None
            implicit_spine = gty.implicit_spine
        case _:
            return None
    if applied >= len(implicit_spine):
        return ()
    return implicit_spine[applied:]
