"""Beta reduction helpers for MLTT terms."""

from __future__ import annotations

from ..ast import (
    App,
    Id,
    IdElim,
    Ctor,
    Elim,
    I,
    Lam,
    Pi,
    Refl,
    Term,
    Univ,
    Var,
)
from ..debruijn import subst


def beta_head_step(t: Term) -> Term:
    match t:
        case App(Lam(_, body), arg):
            return subst(body, arg)
        case App(f, a):
            f1 = beta_head_step(f)
            if f1 != f:
                return App(f1, a)
            return t
        case _:
            return t


def beta_step(term: Term) -> Term:
    """One beta-reduction step anywhere in the term."""

    t1 = beta_head_step(term)
    if t1 != term:
        return t1

    match term:
        case App(f, a):
            f1 = beta_step(f)
            if f1 != f:
                return App(f1, a)
            a1 = beta_step(a)
            if a1 != a:
                return App(f, a1)
            return term

        case Lam(ty, body):
            ty1 = beta_step(ty)
            if ty1 != ty:
                return Lam(ty1, body)
            body1 = beta_step(body)
            if body1 != body:
                return Lam(ty, body1)
            return term

        case Pi(ty, body):
            ty1 = beta_step(ty)
            if ty1 != ty:
                return Pi(ty1, body)
            body1 = beta_step(body)
            if body1 != body:
                return Pi(ty, body1)
            return term

        case Id(ty, l, r):
            ty1 = beta_step(ty)
            if ty1 != ty:
                return Id(ty1, l, r)
            l1 = beta_step(l)
            if l1 != l:
                return Id(ty, l1, r)
            r1 = beta_step(r)
            if r1 != r:
                return Id(ty, l, r1)
            return term

        case Refl(ty, t0):
            ty1 = beta_step(ty)
            if ty1 != ty:
                return Refl(ty1, t0)
            t1 = beta_step(t0)
            if t1 != t0:
                return Refl(ty, t1)
            return term

        case IdElim(A, x, P, d, y, p):
            A1 = beta_step(A)
            if A1 != A:
                return IdElim(A1, x, P, d, y, p)
            x1 = beta_step(x)
            if x1 != x:
                return IdElim(A, x1, P, d, y, p)
            P1 = beta_step(P)
            if P1 != P:
                return IdElim(A, x, P1, d, y, p)
            d1 = beta_step(d)
            if d1 != d:
                return IdElim(A, x, P, d1, y, p)
            y1 = beta_step(y)
            if y1 != y:
                return IdElim(A, x, P, d, y1, p)
            p1 = beta_step(p)
            if p1 != p:
                return IdElim(A, x, P, d, y, p1)
            return term

        case Elim(inductive, motive, cases, scrutinee):
            motive1 = beta_step(motive)
            if motive1 != motive:
                return Elim(inductive, motive1, cases, scrutinee)
            cases1 = [beta_step(branch) for branch in cases]
            if cases1 != cases:
                return Elim(inductive, motive, cases1, scrutinee)
            scrutinee1 = beta_step(scrutinee)
            if scrutinee1 != scrutinee:
                return Elim(inductive, motive, cases, scrutinee1)
            return term

        case Var() | Univ() | I() | Ctor():
            return term

    raise TypeError(f"Unexpected term in beta_step: {term!r}")


__all__ = ["beta_head_step", "beta_step"]
