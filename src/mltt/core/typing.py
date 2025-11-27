"""Type inference and checking for the miniature Martin-Lof type theory."""

from __future__ import annotations

from .ast import (
    App,
    Id,
    IdElim,
    InductiveConstructor,
    InductiveElim,
    InductiveType,
    Lam,
    Pi,
    Refl,
    Term,
    Univ,
    Var,
)
from .debruijn import shift, subst
from .reduce.normalize import normalize


def _apply_term(term: Term, args: tuple[Term, ...]) -> Term:
    result: Term = term
    for arg in args:
        result = App(result, arg)
    return result


def _ctor_type(ctor: InductiveConstructor) -> Term:
    param_count = len(ctor.inductive.param_types)
    index_count = len(ctor.inductive.index_types)
    if len(ctor.result_indices) != index_count:
        raise TypeError("Constructor result indices must match inductive index arity")
    arg_count = len(ctor.arg_types)
    param_vars = tuple(
        Var(arg_count + index_count + param_count - 1 - idx)
        for idx in range(param_count)
    )
    index_vars = tuple(
        Var(arg_count + index_count - 1 - idx) for idx in range(index_count)
    )
    result_indices = tuple(
        _instantiate_params_indices(idx_term, param_vars, index_vars, offset=arg_count)
        for idx_term in ctor.result_indices
    )
    result: Term = _apply_term(ctor.inductive, (*param_vars, *result_indices))

    for arg_ty in reversed(ctor.arg_types):
        result = Pi(arg_ty, result)
    for index_ty in reversed(ctor.inductive.index_types):
        result = Pi(index_ty, result)
    for param_ty in reversed(ctor.inductive.param_types):
        result = Pi(param_ty, result)
    return result


def _apply_ctor(ctor: InductiveConstructor, args: tuple[Term, ...]) -> Term:
    return _apply_term(ctor, args)


def _decompose_app(term: Term) -> tuple[Term, tuple[Term, ...]]:
    args: list[Term] = []
    head = term
    while isinstance(head, App):
        args.insert(0, head.arg)
        head = head.func
    return head, tuple(args)


def _instantiate_params_indices(
    term: Term,
    params: tuple[Term, ...],
    indices: tuple[Term, ...],
    offset: int = 0,
) -> Term:
    """Substitute ``params``/``indices`` (params outermost, indices next)."""
    result = term
    for idx, param in enumerate(reversed(params)):
        result = subst(result, param, j=offset + len(indices) + idx)
    for idx, index in enumerate(reversed(indices)):
        result = subst(result, index, j=offset + idx)
    return result


def _match_inductive_application(
    term: Term, inductive: InductiveType
) -> tuple[tuple[Term, ...], tuple[Term, ...]] | None:
    head, args = _decompose_app(term)
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    total = param_count + index_count
    if head is inductive and len(args) == total:
        return args[:param_count], args[param_count:]
    return None


def _expected_case_type(
    inductive: InductiveType,
    param_args: tuple[Term, ...],
    index_args: tuple[Term, ...],
    motive: Term,
    ctor: InductiveConstructor,
) -> Term:
    binder_roles: list[tuple[str, int | None, Term | None]] = []
    arg_positions: list[int] = []
    instantiated_arg_types = [
        _instantiate_params_indices(arg_ty, param_args, index_args, offset=idx)
        for idx, arg_ty in enumerate(ctor.arg_types)
    ]
    inductive_applied = _apply_term(inductive, (*param_args, *index_args))

    for idx, arg_ty in enumerate(instantiated_arg_types):
        arg_positions.append(len(binder_roles))
        binder_roles.append(("arg", idx, arg_ty))
        match _match_inductive_application(arg_ty, inductive):
            case (ctor_params, _):
                if len(ctor_params) == len(param_args) and all(
                    type_equal(p, a) for p, a in zip(ctor_params, param_args)
                ):
                    binder_roles.append(("ih", idx, None))
            case _:
                pass

    total_binders = len(binder_roles)
    ctor_args = tuple(Var(total_binders - 1 - arg_pos) for arg_pos in arg_positions)
    target: Term = App(
        motive, _apply_ctor(ctor, (*param_args, *index_args, *ctor_args))
    )

    binder_types: list[Term] = []
    for pos, (role, arg_idx, maybe_arg_ty) in enumerate(binder_roles):
        if role == "arg":
            binder_types.append(maybe_arg_ty)  # type: ignore[arg-type]
        else:
            assert arg_idx is not None
            index = pos - 1 - arg_idx
            binder_types.append(App(motive, Var(index)))

    result = target
    for binder_ty in reversed(binder_types):
        result = Pi(binder_ty, result)
    return result


def type_equal(t1: Term, t2: Term) -> bool:
    """Return ``True`` when ``t1`` and ``t2`` normalize to the same term."""

    return normalize(t1) == normalize(t2)


def _extend_ctx(ctx: list[Term], ty: Term) -> list[Term]:
    """Extend ``ctx`` with ``ty`` while keeping indices for outer vars stable."""
    return [shift(ty, 1)] + [shift(x, 1) for x in ctx]


def _expect_universe(term: Term, ctx: list[Term]) -> int:
    """Return the universe level of ``term`` or raise if it is not a type."""

    ty = normalize(infer_type(term, ctx))
    if not isinstance(ty, Univ):
        raise TypeError(f"Expected a universe, got {ty!r}")
    return ty.level


def infer_type(term: Term, ctx: list[Term] | None = None) -> Term:
    """Infer the type of ``term`` under the optional De Bruijn context ``ctx``."""

    ctx = ctx or []
    match term:
        case Var(i):
            if i < len(ctx):
                return ctx[i]
            else:
                raise TypeError(f"Unbound variable {i}")
        case Lam(arg_ty, body):
            body_ty = infer_type(body, _extend_ctx(ctx, arg_ty))
            return Pi(arg_ty, body_ty)
        case App(f, a):
            f_ty = infer_type(f, ctx)
            if not isinstance(f_ty, Pi):
                raise TypeError("Application of non-function")
            if not type_check(a, f_ty.ty, ctx):
                raise TypeError("Function argument type mismatch")
            return subst(f_ty.body, a)
        case Pi(arg_ty, body):
            arg_level = _expect_universe(arg_ty, ctx)
            body_level = _expect_universe(body, _extend_ctx(ctx, arg_ty))
            return Univ(max(arg_level, body_level))
        case Univ(level):
            return Univ(level + 1)
        case InductiveType(param_types, index_types, _, level):
            ctx1 = ctx
            for param_ty in param_types:
                _expect_universe(param_ty, ctx1)
                ctx1 = _extend_ctx(ctx1, param_ty)
            for index_ty in index_types:
                _expect_universe(index_ty, ctx1)
                ctx1 = _extend_ctx(ctx1, index_ty)
            result: Term = Univ(level)
            for index_ty in reversed(index_types):
                result = Pi(index_ty, result)
            for param_ty in reversed(param_types):
                result = Pi(param_ty, result)
            return result
        case InductiveConstructor():
            return _ctor_type(term)
        case InductiveElim(_, motive, _, scrutinee):
            return App(motive, scrutinee)
        case Id(ty, lhs, rhs):
            if not type_check(lhs, ty, ctx) or not type_check(rhs, ty, ctx):
                raise TypeError("Id sides must have given type")
            return Univ(_expect_universe(ty, ctx))
        case Refl(ty, t):
            if not type_check(t, ty, ctx):
                raise TypeError("Refl term not of stated type")
            return Id(ty, t, t)
        case IdElim(A, x, P, d, y, p):
            return App(App(P, y), p)

    raise TypeError(f"Unexpected term in infer_type: {term!r}")


def type_check(term: Term, ty: Term, ctx: list[Term] | None = None) -> bool:
    """Check that ``term`` has type ``ty`` under ``ctx``, raising on mismatches."""

    ctx = ctx or []
    expected_ty = normalize(ty)
    match term:
        case Var(i):
            if i >= len(ctx):
                raise TypeError(f"Unbound variable {i}")
            return type_equal(ctx[i], expected_ty)
        case Lam(arg_ty, body):
            match expected_ty:
                case Pi(dom, cod):
                    if not type_equal(arg_ty, dom):
                        raise TypeError("Lambda domain mismatch")
                    return type_check(body, cod, _extend_ctx(ctx, arg_ty))
                case _:
                    raise TypeError("Lambda expected to have Pi type")
        case App(f, a):
            f_ty = infer_type(f, ctx)
            if not isinstance(f_ty, Pi):
                raise TypeError("Application of non-function")
            if not type_check(a, f_ty.ty, ctx):
                raise TypeError("Application argument type mismatch")
            return type_equal(expected_ty, subst(f_ty.body, a))
        case Pi(_, _):
            return type_equal(expected_ty, infer_type(term, ctx))
        case InductiveType(_, _, _, _):
            return type_equal(expected_ty, infer_type(term, ctx))
        case InductiveConstructor():
            return type_equal(expected_ty, _ctor_type(term))
        case InductiveElim(inductive, motive, cases, scrutinee):
            scrutinee_ty = normalize(infer_type(scrutinee, ctx))
            application = _match_inductive_application(scrutinee_ty, inductive)
            if application is None:
                raise TypeError("InductiveElim scrutinee has wrong type")
            param_args, index_args = application
            inductive_applied = _apply_term(inductive, (*param_args, *index_args))
            if not type_equal(scrutinee_ty, inductive_applied):
                raise TypeError("InductiveElim scrutinee has wrong type")

            motive_ty = infer_type(motive, ctx)
            if not isinstance(motive_ty, Pi):
                raise TypeError("InductiveElim motive not a function")
            if not type_equal(motive_ty.ty, inductive_applied):
                raise TypeError("InductiveElim motive domain mismatch")
            motive_level = _expect_universe(
                motive_ty.body, _extend_ctx(ctx, inductive_applied)
            )

            if len(cases) != len(inductive.constructors):
                raise TypeError("InductiveElim cases do not match constructors")

            for ctor, branch in zip(inductive.constructors, cases):
                branch_ty = _expected_case_type(
                    inductive, param_args, index_args, motive, ctor
                )
                if not type_check(branch, branch_ty, ctx):
                    raise TypeError("Case for constructor has wrong type")

            target_ty = App(motive, scrutinee)
            target_level = _expect_universe(target_ty, ctx)
            if target_level > motive_level:
                raise TypeError("InductiveElim motive returns too small a universe")
            return type_equal(expected_ty, target_ty)
        case Id(id_ty, l, r):
            if not type_check(l, id_ty, ctx) or not type_check(r, id_ty, ctx):
                raise TypeError("Id sides not of given type")
            return isinstance(expected_ty, Univ)
        case Refl(rty, t):
            if not type_check(t, rty, ctx):
                raise TypeError("Refl term not of stated type")
            return type_equal(expected_ty, Id(rty, t, t))
        case IdElim(A, x, P, d, y, p):
            if not type_check(x, A, ctx):
                raise TypeError("IdElim: x : A fails")
            if not type_check(y, A, ctx):
                raise TypeError("IdElim: y : A fails")
            if not type_check(p, Id(A, x, y), ctx):
                raise TypeError("IdElim: p : Id(A,x,y) fails")
            if not type_check(d, App(App(P, x), Refl(A, x)), ctx):
                raise TypeError("IdElim: d : P x (Refl x) fails")
            return type_equal(expected_ty, App(App(P, y), p))
        case Univ(_):
            return isinstance(expected_ty, Univ)

    raise TypeError(f"Unexpected term in type_check: {term!r}")


__all__ = ["type_equal", "infer_type", "type_check"]
