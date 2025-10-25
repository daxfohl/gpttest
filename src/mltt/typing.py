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
    Pair,
    Pi,
    Refl,
    Sigma,
    Succ,
    Term,
    TypeUniverse,
    Var,
    Zero,
)
from .eval import normalize
from .predicates import is_nat_type, is_pi, is_sigma, is_type_universe


def type_equal(t1: Term, t2: Term) -> bool:
    """Return ``True`` when ``t1`` and ``t2`` normalize to the same term."""

    return normalize(t1) == normalize(t2)


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
            body_ty = infer_type(body, [arg_ty] + ctx)
            return Pi(arg_ty, body_ty)
        case App(f, a):
            f_ty = infer_type(f, ctx)
            if not is_pi(f_ty):
                raise TypeError("Application of non-function")
            if not type_check(a, f_ty.ty, ctx):
                raise TypeError("Function argument type mismatch")
            return subst(f_ty.body, a)
        case Pi(_, _):
            return TypeUniverse()
        case Sigma(_, _):
            return TypeUniverse()
        case Pair(_, _):
            raise TypeError("Cannot infer type of Pair without expected Sigma")
        case TypeUniverse():
            return TypeUniverse()
        case NatType():
            return TypeUniverse()
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
            return TypeUniverse()
        case Refl(ty, t):
            if not type_check(t, ty, ctx):
                raise TypeError("Refl term not of stated type")
            return Id(ty, t, t)
        case IdElim(A, x, P, d, y, p):
            return App(App(P, y), p)
        case _:
            raise TypeError(f"Cannot infer type of {term}")


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
                    return type_check(body, cod, [arg_ty] + ctx)
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
            return is_type_universe(expected_ty)
        case Sigma(_, _):
            return is_type_universe(expected_ty)
        case Pair(fst, snd):
            if not is_sigma(expected_ty):
                raise TypeError("Pair expected to have Sigma type")
            ok1 = type_check(fst, expected_ty.ty, ctx)
            ok2 = type_check(snd, subst(expected_ty.body, fst), ctx)
            return ok1 and ok2
        case TypeUniverse():
            return is_type_universe(expected_ty)
        case NatType():
            return is_type_universe(expected_ty)
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
            step_ty = Pi(NatType(), Pi(App(P, Var(0)), App(P, Succ(Var(0)))))
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
        case _:
            inferred = infer_type(term, ctx)
            return type_equal(inferred, expected_ty)


from .debruijn import subst

__all__ = ["type_equal", "infer_type", "type_check"]
