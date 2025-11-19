"""Beta reduction and evaluation helpers for MLTT terms."""

from __future__ import annotations

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
from .debruijn import subst


def beta_head_step(t: Term) -> Term:
    match t:
        case App(Lam(_, body), arg):
            return subst(body, arg)
        case App(f, a):
            f1 = beta_head_step(f)
            if f1 != f:
                return App(f1, a)
            return t
        # don’t go under Lam/body, don’t touch arguments further
        case _:
            return t


def beta_step(term: Term) -> Term:
    """One beta-reduction step anywhere in the term.

    Prefer head β via beta_head_step; if none, recurse into subterms.
    """

    # 1. Try a head β step first
    t1 = beta_head_step(term)
    if t1 != term:
        return t1

    # 2. No head β redex; search inside
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

        case NatRec(P, z, s, n):
            P1 = beta_step(P)
            if P1 != P:
                return NatRec(P1, z, s, n)
            z1 = beta_step(z)
            if z1 != z:
                return NatRec(P, z1, s, n)
            s1 = beta_step(s)
            if s1 != s:
                return NatRec(P, z, s1, n)
            n1 = beta_step(n)
            if n1 != n:
                return NatRec(P, z, s, n1)
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

        case Succ(n):
            n1 = beta_step(n)
            if n1 != n:
                return Succ(n1)
            return term

        case Var(_) | NatType() | Univ() | Zero():
            return term

    raise TypeError(f"Unexpected term in beta_step: {term!r}")


def iota_head_step(t: Term) -> Term:
    match t:
        case NatRec(P, z, s, Zero()):
            return z
        case NatRec(P, z, s, Succ(k)):
            return App(App(s, k), NatRec(P, z, s, k))
        case IdElim(A, x, P, d, y, Refl(_, _)):
            return d
        case NatRec(P, z, s, n):
            n1 = iota_head_step(n)
            if n1 != n:
                return NatRec(P, z, s, n1)
            return t
        case IdElim(A, x, P, d, y, p):
            p1 = iota_head_step(p)
            if p1 != p:
                return IdElim(A, x, P, d, y, p1)
            return t
        case _:
            return t


def iota_step(term: Term) -> Term:
    """One iota-reduction step anywhere (NatRec / IdElim)."""

    # 1. Try a head ι step first
    t1 = iota_head_step(term)
    if t1 != term:
        return t1

    # 2. No head ι redex; search inside
    match term:
        case App(f, a):
            f1 = iota_step(f)
            if f1 != f:
                return App(f1, a)
            a1 = iota_step(a)
            if a1 != a:
                return App(f, a1)
            return term

        case Lam(ty, body):
            ty1 = iota_step(ty)
            if ty1 != ty:
                return Lam(ty1, body)
            body1 = iota_step(body)
            if body1 != body:
                return Lam(ty, body1)
            return term

        case Pi(ty, body):
            ty1 = iota_step(ty)
            if ty1 != ty:
                return Pi(ty1, body)
            body1 = iota_step(body)
            if body1 != body:
                return Pi(ty, body1)
            return term

        case NatRec(P, z, s, n):
            P1 = iota_step(P)
            if P1 != P:
                return NatRec(P1, z, s, n)
            z1 = iota_step(z)
            if z1 != z:
                return NatRec(P, z1, s, n)
            s1 = iota_step(s)
            if s1 != s:
                return NatRec(P, z, s1, n)
            n1 = iota_step(n)
            if n1 != n:
                return NatRec(P, z, s, n1)
            return term

        case Id(ty, l, r):
            ty1 = iota_step(ty)
            if ty1 != ty:
                return Id(ty1, l, r)
            l1 = iota_step(l)
            if l1 != l:
                return Id(ty, l1, r)
            r1 = iota_step(r)
            if r1 != r:
                return Id(ty, l, r1)
            return term

        case Refl(ty, t0):
            ty1 = iota_step(ty)
            if ty1 != ty:
                return Refl(ty1, t0)
            t1 = iota_step(t0)
            if t1 != t0:
                return Refl(ty, t1)
            return term

        case IdElim(A, x, P, d, y, p):
            A1 = iota_step(A)
            if A1 != A:
                return IdElim(A1, x, P, d, y, p)
            x1 = iota_step(x)
            if x1 != x:
                return IdElim(A, x1, P, d, y, p)
            P1 = iota_step(P)
            if P1 != P:
                return IdElim(A, x, P1, d, y, p)
            d1 = iota_step(d)
            if d1 != d:
                return IdElim(A, x, P, d1, y, p)
            y1 = iota_step(y)
            if y1 != y:
                return IdElim(A, x, P, d, y1, p)
            p1 = iota_step(p)
            if p1 != p:
                return IdElim(A, x, P, d, y, p1)
            return term

        case Succ(n):
            n1 = iota_step(n)
            if n1 != n:
                return Succ(n1)
            return term

        case Var(_) | NatType() | Univ() | Zero():
            return term

    raise TypeError(f"Unexpected term in iota_step: {term!r}")


def whnf_step(term: Term) -> Term:
    """One small-step using beta or iota."""
    t1 = beta_head_step(term)
    if t1 != term:
        return t1
    t2 = iota_head_step(term)
    if t2 != term:
        return t2
    return term


def whnf(term: Term) -> Term:
    while True:
        t1 = whnf_step(term)
        if t1 == term:
            return term
        term = t1


def normalize_step(term: Term) -> Term:
    """One small-step using beta or iota."""
    t1 = beta_step(term)
    if t1 != term:
        return t1
    t2 = iota_step(term)
    if t2 != term:
        return t2
    return term


def normalize(term: Term) -> Term:
    """Normalize ``term`` by repeatedly reducing until no rules apply."""
    while True:
        t1 = normalize_step(term)
        if t1 == term:
            return term
        term = t1


__all__ = ["normalize_step", "whnf", "normalize"]
