"""Beta reduction helpers for MLTT terms."""

from __future__ import annotations

from ..ast import App, Ctor, Elim, I, Lam, Pi, Term, Univ, Var
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

        case Elim(inductive, motive, cases, scrutinee):
            motive1 = beta_step(motive)
            if motive1 != motive:
                return Elim(inductive, motive1, cases, scrutinee)
            cases1 = tuple(beta_step(case) for case in cases)
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
