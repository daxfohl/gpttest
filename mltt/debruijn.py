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
)


def shift(term: Term, by: int, cutoff: int = 0) -> Term:
    match term:
        case Var(index):
            return Var(index + by if index >= cutoff else index)
        case Lam(ty, body):
            return Lam(shift(ty, by, cutoff), shift(body, by, cutoff + 1))
        case Pi(ty, body):
            return Pi(shift(ty, by, cutoff), shift(body, by, cutoff + 1))
        case Sigma(ty, body):
            return Sigma(shift(ty, by, cutoff), shift(body, by, cutoff + 1))
        case Pair(fst, snd):
            return Pair(shift(fst, by, cutoff), shift(snd, by, cutoff))
        case App(f, a):
            return App(shift(f, by, cutoff), shift(a, by, cutoff))
        case NatRec(P, z, s, n):
            return NatRec(
                shift(P, by, cutoff),
                shift(z, by, cutoff),
                shift(s, by, cutoff),
                shift(n, by, cutoff),
            )
        case Succ(n):
            return Succ(shift(n, by, cutoff))
        case Id(ty, l, r):
            return Id(shift(ty, by, cutoff), shift(l, by, cutoff), shift(r, by, cutoff))
        case Refl(ty, t):
            return Refl(shift(ty, by, cutoff), shift(t, by, cutoff))
        case IdElim(A, x, P, d, y, p):
            return IdElim(
                shift(A, by, cutoff),
                shift(x, by, cutoff),
                shift(P, by, cutoff),
                shift(d, by, cutoff),
                shift(y, by, cutoff),
                shift(p, by, cutoff),
            )
        case _:
            return term


def subst(term: Term, sub: Term, depth: int = 0) -> Term:
    match term:
        case Var(index):
            if index == depth:
                return shift(sub, 0, depth)
            elif index > depth:
                return Var(index - 1)
            else:
                return term
        case Lam(ty, body):
            return Lam(subst(ty, sub, depth), subst(body, shift(sub, 1), depth + 1))
        case Pi(ty, body):
            return Pi(subst(ty, sub, depth), subst(body, shift(sub, 1), depth + 1))
        case Sigma(ty, body):
            return Sigma(subst(ty, sub, depth), subst(body, shift(sub, 1), depth + 1))
        case Pair(fst, snd):
            return Pair(subst(fst, sub, depth), subst(snd, sub, depth))
        case App(f, a):
            return App(subst(f, sub, depth), subst(a, sub, depth))
        case NatRec(P, z, s, n):
            return NatRec(
                subst(P, sub, depth),
                subst(z, sub, depth),
                subst(s, sub, depth),
                subst(n, sub, depth),
            )
        case Succ(n):
            return Succ(subst(n, sub, depth))
        case Id(ty, l, r):
            return Id(subst(ty, sub, depth), subst(l, sub, depth), subst(r, sub, depth))
        case Refl(ty, t):
            return Refl(subst(ty, sub, depth), subst(t, sub, depth))
        case IdElim(A, x, P, d, y, p):
            return IdElim(
                subst(A, sub, depth),
                subst(x, sub, depth),
                subst(P, sub, depth),
                subst(d, sub, depth),
                subst(y, sub, depth),
                subst(p, sub, depth),
            )
        case _:
            return term


__all__ = ["shift", "subst"]
