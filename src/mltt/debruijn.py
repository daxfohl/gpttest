"""Utilities for working with De Bruijn indices such as shifting and substitution."""

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
    """Shift free variables in ``term`` by ``by`` starting at ``cutoff``."""

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

def subst_impl(term: Term, sub: Term, depth: int) -> Term:
    """Substitute ``sub`` for the variable at ``depth`` inside ``term``."""
    match term:
        case Var(index):
            if index == depth:
                return sub
            else:
                return Var(index)  # ← no decrement here

        case Lam(ty, body):
            return Lam(
                subst_impl(ty, sub, depth),
                subst_impl(body, shift(sub, 1, 0), depth + 1),
            )

        case Pi(ty, body):
            return Pi(
                subst_impl(ty, sub, depth),
                subst_impl(body, shift(sub, 1, 0), depth + 1),
            )

        case Sigma(ty, body):
            return Sigma(
                subst_impl(ty, sub, depth),
                subst_impl(body, shift(sub, 1, 0), depth + 1),
            )

        case Pair(fst, snd):
            return Pair(subst_impl(fst, sub, depth), subst_impl(snd, sub, depth))

        case App(f, a):
            return App(subst_impl(f, sub, depth), subst_impl(a, sub, depth))

        case NatRec(P, z, s, n):
            return NatRec(
                subst_impl(P, sub, depth),
                subst_impl(z, sub, depth),
                subst_impl(s, sub, depth),
                subst_impl(n, sub, depth),
            )

        case Succ(n):
            return Succ(subst_impl(n, sub, depth))

        case Id(ty, l, r):
            return Id(
                subst_impl(ty, sub, depth),
                subst_impl(l, sub, depth),
                subst_impl(r, sub, depth),
            )

        case Refl(ty, t):
            return Refl(subst_impl(ty, sub, depth), subst_impl(t, sub, depth))

        case IdElim(A, x, P, d, y, p):
            return IdElim(
                subst_impl(A, sub, depth),
                subst_impl(x, sub, depth),
                subst_impl(P, sub, depth),
                subst_impl(d, sub, depth),
                subst_impl(y, sub, depth),
                subst_impl(p, sub, depth),
            )

        case _:
            return term


def subst(term: Term, sub: Term) -> Term:
    # TAPL-style safe substitution for topmost var:
    return shift(subst_impl(term, shift(sub, 1, 0), 0), -1, 0)



__all__ = ["subst"]
