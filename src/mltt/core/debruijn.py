"""Utilities for working with De Bruijn indices such as shifting and substitution."""

from __future__ import annotations

from .ast import (
    App,
    InductiveConstructor,
    Id,
    IdElim,
    InductiveElim,
    InductiveType,
    Lam,
    Pi,
    Refl,
    Term,
    Univ,
    Var,
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
        case InductiveElim(inductive, motive, cases, scrutinee):
            return InductiveElim(
                inductive,
                shift(motive, by, cutoff),
                [shift(branch, by, cutoff) for branch in cases],
                shift(scrutinee, by, cutoff),
            )
        case InductiveConstructor():
            return term
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
        case Univ() | InductiveType():
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
        case InductiveElim(inductive, motive, cases, scrutinee):
            return InductiveElim(
                inductive,
                subst(motive, sub, j),
                [subst(branch, sub, j) for branch in cases],
                subst(scrutinee, sub, j),
            )
        case InductiveConstructor():
            return term
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
        case Univ() | InductiveType():
            return term

    raise TypeError(f"Unexpected term in subst: {term!r}")


__all__ = ["subst"]
