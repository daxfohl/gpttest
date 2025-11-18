"""Type inference and checking for the miniature Martin-Lof type theory."""

from __future__ import annotations

from typing import List, Optional

from .ast import (
    App,
    Id,
    IdElim,
    Lam,
    NatRec,
    NatType,
    Pi,
    Refl,
    Succ,
    Term,
    Univ,
    Var,
    Zero,
)
from .beta_reduce import normalize
from .predicates import is_nat_type, is_pi, is_type_universe


def type_equal(t1: Term, t2: Term) -> bool:
    """Return ``True`` when ``t1`` and ``t2`` normalize to the same term."""

    return normalize(t1) == normalize(t2)


def _extend_ctx(ctx: List[Term], ty: Term) -> List[Term]:
    """Extend ``ctx`` with ``ty`` while keeping indices for outer vars stable."""

    return [shift(ty, 1)] + ctx


def _expect_universe(term: Term, ctx: List[Term]) -> int:
    """Return the universe level of ``term`` or raise if it is not a type."""

    ty = normalize(infer_type(term, ctx))
    match ty:
        case Univ(level):
            return level
        case _:
            raise TypeError(f"Expected a universe, got {ty!r}")


def infer_type(term: Term, ctx: Optional[List[Term]] = None) -> Term:
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
            if not is_pi(f_ty):
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
        case NatType():
            return Univ(0)
        case Zero():
            return NatType()
        case Succ(n):
            if not type_check(n, NatType(), ctx):
                raise TypeError("Succ expects Nat")
            return NatType()
        case NatRec(P, z, s, n):
            return App(P, n)
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


def type_check(term: Term, ty: Term, ctx: Optional[List[Term]] = None) -> bool:
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
            if not is_pi(f_ty):
                raise TypeError("Application of non-function")
            if not type_check(a, f_ty.ty, ctx):
                raise TypeError("Application argument type mismatch")
            return type_equal(expected_ty, subst(f_ty.body, a))
        case Pi(_, _):
            return type_equal(expected_ty, infer_type(term, ctx))
        case Zero():
            if not is_nat_type(expected_ty):
                raise TypeError("Zero must have type Nat")
            return True
        case Succ(n):
            if not is_nat_type(expected_ty):
                raise TypeError("Succ must have type Nat")
            return type_check(n, NatType(), ctx)
        case NatRec(P, z, s, n):
            if not type_check(n, NatType(), ctx):
                raise TypeError("NatRec scrutinee not Nat")
            if not type_check(z, App(P, Zero()), ctx):
                raise TypeError("NatRec base case type mismatch (z : P 0)")
            step_ty = Pi(NatType(), Pi(App(P, Var(0)), App(P, Succ(Var(1)))))
            if not type_check(s, step_ty, ctx):
                raise TypeError("NatRec step case type mismatch")
            return type_equal(expected_ty, App(P, n))
        case Id(id_ty, l, r):
            if not type_check(l, id_ty, ctx) or not type_check(r, id_ty, ctx):
                raise TypeError("Id sides not of given type")
            return is_type_universe(expected_ty)
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
        case NatType() | Univ(_):
            return is_type_universe(expected_ty)

    raise TypeError(f"Unexpected term in type_check: {term!r}")


from .debruijn import shift, subst

__all__ = ["type_equal", "infer_type", "type_check"]
