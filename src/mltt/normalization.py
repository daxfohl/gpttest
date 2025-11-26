"""Beta reduction and evaluation helpers for MLTT terms."""

from __future__ import annotations

from typing import Mapping, Sequence

from .ast import (
    App,
    Id,
    IdElim,
    InductiveConstructor,
    InductiveElim,
    InductiveType,
    Lam,
    Pi,
    Refl,
    Term,
    Univ,
    Var,
)
from .debruijn import subst


def _apply_ctor(ctor: InductiveConstructor, args: Sequence[Term]) -> Term:
    term: Term = ctor
    for arg in args:
        term = App(term, arg)
    return term


def _decompose_ctor_app(
    term: Term,
) -> tuple[InductiveConstructor, tuple[Term, ...]] | None:
    args: list[Term] = []
    head = term
    while isinstance(head, App):
        args.insert(0, head.arg)
        head = head.func
    if isinstance(head, InductiveConstructor):
        return head, tuple(args)
    return None


def _ensure_constructor(inductive: InductiveType, ctor: InductiveConstructor) -> None:
    if not any(ctor is ctor_def for ctor_def in inductive.constructors):
        raise TypeError("Constructor does not belong to inductive type")


def _iota_constructor(
    inductive: InductiveType,
    motive: Term,
    cases: Mapping[InductiveConstructor, Term],
    ctor: InductiveConstructor,
    args: Sequence[Term],
) -> Term:
    _ensure_constructor(inductive, ctor)
    branch = cases.get(ctor)
    if branch is None:
        return InductiveElim(inductive, motive, cases, _apply_ctor(ctor, args))

    applied_args: list[Term] = []
    for arg_ty, arg in zip(ctor.arg_types, args, strict=False):
        applied_args.append(arg)
        if isinstance(arg_ty, InductiveType) and arg_ty is inductive:
            applied_args.append(InductiveElim(inductive, motive, cases, arg))

    result: Term = branch
    for a in applied_args:
        result = App(result, a)
    return result


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

        case InductiveElim(inductive, motive, cases, scrutinee):
            motive1 = beta_step(motive)
            if motive1 != motive:
                return InductiveElim(inductive, motive1, cases, scrutinee)
            cases1 = {k: beta_step(v) for k, v in cases.items()}
            if list(cases1.items()) != list(cases.items()):
                return InductiveElim(inductive, motive, cases1, scrutinee)
            scrutinee1 = beta_step(scrutinee)
            if scrutinee1 != scrutinee:
                return InductiveElim(inductive, motive, cases, scrutinee1)
            return term

        case Var(_) | Univ() | InductiveType() | InductiveConstructor():
            return term

    raise TypeError(f"Unexpected term in beta_step: {term!r}")


def iota_head_step(t: Term) -> Term:
    match t:
        case InductiveElim(inductive, motive, cases, scrutinee):
            decomposition = _decompose_ctor_app(scrutinee)
            if decomposition:
                ctor, args = decomposition
                if ctor.inductive is inductive and len(args) == len(ctor.arg_types):
                    return _iota_constructor(inductive, motive, cases, ctor, args)
            scrutinee1 = iota_head_step(scrutinee)
            if scrutinee1 != scrutinee:
                return InductiveElim(inductive, motive, cases, scrutinee1)
            return t
        case IdElim(A, x, P, d, y, Refl(_, _)):
            return d
        case IdElim(A, x, P, d, y, p):
            p1 = iota_head_step(p)
            if p1 != p:
                return IdElim(A, x, P, d, y, p1)
            return t
        case _:
            return t


def iota_step(term: Term) -> Term:
    """One iota-reduction step anywhere (InductiveElim / IdElim)."""

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

        case InductiveElim(inductive, motive, cases, scrutinee):
            motive1 = iota_step(motive)
            if motive1 != motive:
                return InductiveElim(inductive, motive1, cases, scrutinee)
            cases1 = {k: iota_step(v) for k, v in cases.items()}
            if list(cases1.items()) != list(cases.items()):
                return InductiveElim(inductive, motive, cases1, scrutinee)
            scrutinee1 = iota_step(scrutinee)
            if scrutinee1 != scrutinee:
                return InductiveElim(inductive, motive, cases, scrutinee1)
            return term

        case Var(_) | Univ() | InductiveType() | InductiveConstructor():
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
