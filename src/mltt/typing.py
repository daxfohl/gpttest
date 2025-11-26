"""Type inference and checking for the miniature Martin-Lof type theory."""

from __future__ import annotations

from .ast import (
    App,
    ConstructorApp,
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
from .normalization import normalize


def _ensure_constructor(inductive: InductiveType, ctor: InductiveConstructor) -> None:
    if not any(ctor is ctor_def for ctor_def in inductive.constructors):
        raise TypeError("Constructor does not belong to inductive type")


def _expected_case_type(
    inductive: InductiveType, motive: Term, ctor: InductiveConstructor
) -> Term:
    binder_roles: list[tuple[str, int | None, Term | None]] = []
    arg_positions: list[int] = []

    for idx, arg_ty in enumerate(ctor.arg_types):
        arg_positions.append(len(binder_roles))
        binder_roles.append(("arg", idx, arg_ty))
        if isinstance(arg_ty, InductiveType) and arg_ty is inductive:
            binder_roles.append(("ih", idx, None))

    total_binders = len(binder_roles)
    ctor_args = tuple(Var(total_binders - 1 - arg_pos) for arg_pos in arg_positions)
    target: Term = App(motive, ConstructorApp(inductive, ctor, ctor_args))

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
        case InductiveType(_, level):
            return Univ(level)
        case ConstructorApp(ind, ctor, args):
            _ensure_constructor(ind, ctor)
            if len(args) != len(ctor.arg_types):
                raise TypeError("Constructor applied to wrong number of arguments")
            for arg, arg_ty in zip(args, ctor.arg_types, strict=False):
                if not type_check(arg, arg_ty, ctx):
                    raise TypeError("Constructor argument fails type check")
            return ind
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
        case InductiveType(_, level):
            return isinstance(expected_ty, Univ) and expected_ty.level >= level
        case ConstructorApp(ind, ctor, args):
            if not type_equal(expected_ty, ind):
                raise TypeError("Constructor must have inductive type")
            _ensure_constructor(ind, ctor)
            if len(args) != len(ctor.arg_types):
                raise TypeError("Constructor applied to wrong number of arguments")
            for arg, arg_ty in zip(args, ctor.arg_types, strict=False):
                if not type_check(arg, arg_ty, ctx):
                    raise TypeError("Constructor argument fails type check")
            return True
        case InductiveElim(inductive, motive, cases, scrutinee):
            if not type_check(scrutinee, inductive, ctx):
                raise TypeError("InductiveElim scrutinee has wrong type")

            motive_ty = infer_type(motive, ctx)
            if not isinstance(motive_ty, Pi):
                raise TypeError("InductiveElim motive not a function")
            if not type_equal(motive_ty.ty, inductive):
                raise TypeError("InductiveElim motive domain mismatch")
            motive_level = _expect_universe(motive_ty.body, _extend_ctx(ctx, inductive))

            if set(cases.keys()) != set(inductive.constructors):
                raise TypeError("InductiveElim cases do not match constructors")

            for ctor in inductive.constructors:
                branch_ty = _expected_case_type(inductive, motive, ctor)
                branch = cases.get(ctor)
                assert branch is not None
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
