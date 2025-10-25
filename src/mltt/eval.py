"""Beta reduction and evaluation helpers for MLTT terms."""

from __future__ import annotations

from .ast import (
    App,
    Id,
    IdElim,
    Lam,
    NatRec,
    Pair,
    Pi,
    Refl,
    Sigma,
    Succ,
    Term,
    Var,
    Zero,
)
from .debruijn import subst


def beta_reduce(term: Term) -> Term:
    """Fully beta-reduce ``term`` by recursively rewriting every redex."""

    match term:
        case App(Lam(_, body), arg):
            return beta_reduce(subst(body, arg))
        case App(f, a):
            return App(beta_reduce(f), beta_reduce(a))
        case Lam(ty, body):
            return Lam(beta_reduce(ty), beta_reduce(body))
        case Pi(ty, body):
            return Pi(beta_reduce(ty), beta_reduce(body))
        case Sigma(ty, body):
            return Sigma(beta_reduce(ty), beta_reduce(body))
        case Pair(fst, snd):
            return Pair(beta_reduce(fst), beta_reduce(snd))
        case Succ(n):
            return Succ(beta_reduce(n))
        case NatRec(P, z, s, n):
            return NatRec(beta_reduce(P), beta_reduce(z), beta_reduce(s), beta_reduce(n))
        case Id(ty, l, r):
            return Id(beta_reduce(ty), beta_reduce(l), beta_reduce(r))
        case Refl(ty, t):
            return Refl(beta_reduce(ty), beta_reduce(t))
        case IdElim(A, x, P, d, y, p):
            return IdElim(
                beta_reduce(A),
                beta_reduce(x),
                beta_reduce(P),
                beta_reduce(d),
                beta_reduce(y),
                beta_reduce(p),
            )
        case _:
            return term


def whnf(term: Term) -> Term:
    """Compute the weak head normal form of ``term`` without normalizing subterms."""

    match term:
        case App(Lam(_, body), arg):
            return whnf(subst(body, arg))
        case App(f, a):
            f_wh = whnf(f)
            if f_wh != f:
                return whnf(App(f_wh, a))
            return App(f_wh, a)
        case NatRec(P, z, s, n):
            n_wh = whnf(n)
            match n_wh:
                case Zero():
                    return z
                case Succ(k):
                    return App(App(s, k), NatRec(P, z, s, k))
                case _:
                    return NatRec(P, z, s, n_wh)
        case IdElim(A, x, P, d, y, p):
            p_wh = whnf(p)
            match p_wh:
                case Refl(_, _):
                    return d
                case _:
                    return IdElim(A, x, P, d, y, p_wh)
        case _:
            return term


def beta_step(term: Term) -> Term:
    """Perform a single beta-reduction step on ``term`` if possible."""

    match term:
        case App(Lam(_, body), arg):
            return subst(body, arg)
        case App(f, a):
            f1 = beta_step(f)
            if f1 != f:
                return App(f1, a)
            a1 = beta_step(a)
            if a1 != a:
                return App(f, a1)
            return term
        case Lam(ty, body):
            body1 = beta_step(body)
            if body1 != body:
                return Lam(ty, body1)
            return term
        case NatRec(P, z, s, n):
            match n:
                case Zero():
                    return z
                case Succ(k):
                    return App(App(s, k), NatRec(P, z, s, k))
                case _:
                    n1 = beta_step(n)
                    if n1 != n:
                        return NatRec(P, z, s, n1)
                    return term
        case IdElim(A, x, P, d, y, p):
            match p:
                case Refl(_, _):
                    return d
                case _:
                    p1 = beta_step(p)
                    if p1 != p:
                        return IdElim(A, x, P, d, y, p1)
                    return term
        case Succ(n):
            n1 = beta_step(n)
            if n1 != n:
                return Succ(n1)
            return term
        case _:
            return term


def normalize(term: Term) -> Term:
    """Normalize ``term`` by repeatedly reducing until no rules apply."""

    match term:
        case Var(_):
            return term
        case Lam(ty, body):
            return Lam(normalize(ty), normalize(body))
        case Pi(ty, body):
            return Pi(normalize(ty), normalize(body))
        case Sigma(ty, body):
            return Sigma(normalize(ty), normalize(body))
        case Pair(fst, snd):
            return Pair(normalize(fst), normalize(snd))
        case App(f, a):
            f_n = normalize(f)
            a_n = normalize(a)
            match f_n:
                case Lam(_, body):
                    return normalize(subst(body, a_n))
                case _:
                    return App(f_n, a_n)
        case Zero():
            return Zero()
        case Succ(n):
            return Succ(normalize(n))
        case NatRec(P, z, s, n):
            n_val = normalize(n)
            match n_val:
                case Zero():
                    return normalize(z)
                case Succ(k):
                    ih = normalize(NatRec(P, z, s, k))
                    return normalize(App(App(s, k), ih))
                case _:
                    return NatRec(normalize(P), normalize(z), normalize(s), n_val)
        case Id(ty, l, r):
            return Id(normalize(ty), normalize(l), normalize(r))
        case Refl(ty, t):
            return Refl(normalize(ty), normalize(t))
        case IdElim(A, x, P, d, y, p):
            p_n = normalize(p)
            match p_n:
                case Refl(_, _):
                    return normalize(d)
                case _:
                    return IdElim(
                        normalize(A),
                        normalize(x),
                        normalize(P),
                        normalize(d),
                        normalize(y),
                        p_n,
                    )
        case _:
            return term


__all__ = ["beta_reduce", "whnf", "beta_step", "normalize"]
