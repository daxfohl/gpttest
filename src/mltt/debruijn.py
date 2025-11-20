"""Utilities for working with De Bruijn indices such as shifting and substitution."""

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


def shift(term: Term, by: int, cutoff: int = 0) -> Term:
    """Shift free variables in ``term`` by ``by`` starting at ``cutoff``."""

    match term:
        case Var(k):
            return Var(k + by if k >= cutoff else k)
        case Lam(ty, body):
            return Lam(shift(ty, by, cutoff), shift(body, by, cutoff + 1))
        case Pi(ty, body):
            return Pi(shift(ty, by, cutoff), shift(body, by, cutoff + 1))
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
        case Univ() | NatType() | Zero():
            return term

    raise TypeError(f"Unexpected term in shift: {term!r}")


def subst(term: Term, sub: Term, j: int = 0) -> Term:
    """Substitute ``sub`` for ``Var(j)`` inside ``term``, and squash it."""
    match term:
        case Var(k):
            if k == j:
                return sub
            elif k > j:
                return Var(k - 1)
            else:
                return term
        case Lam(ty, body):
            return Lam(
                subst(ty, sub, j),
                subst(body, shift(sub, 1, 0), j + 1),
            )
        case Pi(ty, body):
            return Pi(
                subst(ty, sub, j),
                subst(body, shift(sub, 1, 0), j + 1),
            )
        case App(f, a):
            return App(subst(f, sub, j), subst(a, sub, j))
        case NatRec(P, z, s, n):
            return NatRec(
                subst(P, sub, j),
                subst(z, sub, j),
                subst(s, sub, j),
                subst(n, sub, j),
            )
        case Succ(n):
            return Succ(subst(n, sub, j))
        case Id(ty, l, r):
            return Id(
                subst(ty, sub, j),
                subst(l, sub, j),
                subst(r, sub, j),
            )
        case Refl(ty, t):
            return Refl(subst(ty, sub, j), subst(t, sub, j))
        case IdElim(A, x, P, d, y, p):
            return IdElim(
                subst(A, sub, j),
                subst(x, sub, j),
                subst(P, sub, j),
                subst(d, sub, j),
                subst(y, sub, j),
                subst(p, sub, j),
            )
        case Univ() | NatType() | Zero():
            return term

    raise TypeError(f"Unexpected term in subst: {term!r}")


__all__ = ["subst"]
