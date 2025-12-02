"""Weak head normal form reduction leveraging beta/iota head steps."""

from __future__ import annotations

from typing import Callable

from ..ast import Term, App, Lam, InductiveElim, InductiveConstructor, InductiveType, IdElim, Refl, Var, Univ, Id, Pi
from ..debruijn import subst
from ..inductive_utils import decompose_ctor_app, ctor_index, decompose_app, apply_term


def iota_reduce(
    ctor: InductiveConstructor,
    cases: list[Term],
    args: tuple[Term, ...],
    motive: Term,
) -> Term:
    """Compute the iota-reduction of an eliminator on a fully-applied ctor."""
    inductive = ctor.inductive
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    index = ctor_index(ctor)
    branch = cases[index]
    ctor_args = args[param_count + index_count :]

    ihs: list[Term] = []
    for arg_term, arg_ty in zip(ctor_args, ctor.arg_types, strict=True):
        head, head_args = decompose_app(arg_ty)
        if head is ctor.inductive:
            # only works if after substituting param_args and index_args into ctor_arg_types.
            # assert head_args[:param_count] == param_args, f"{arg_ty}: {head_args[:param_count]!r} == {param_args}"
            ih = InductiveElim(
                inductive=ctor.inductive,
                motive=motive,
                cases=cases,
                scrutinee=arg_term,
            )
            ihs.append(ih)

    all_args = (*ctor_args, *ihs)
    test = apply_term(branch, all_args)
    return test


def whnf(term: Term) -> Term:
    match term:
        case App(f, a):
            f_whnf = whnf(f)
            match f_whnf:
                case Lam(_, body):
                    return whnf(subst(body, a))
                case _:
                    return App(f_whnf, a)

        case InductiveElim(params, motive, cases, scrutinee):
            scrutinee_whnf = whnf(scrutinee)
            match decompose_ctor_app(scrutinee_whnf):
                case (ctor, args):
                    return whnf(iota_reduce(ctor, cases, args, motive))
                case _:
                    # Decomposition terminates in Var or Axiom, etc.
                    return InductiveElim(params, motive, cases, scrutinee_whnf)

        case IdElim(A, x, P, d, y, Refl(_, _)):
            return d
        case IdElim(A, x, P, d, y, p):
            # Push reduction into the proof if the head does not expose a Refl.
            p1 = whnf(p)
            if p1 != p:
                return IdElim(A, x, P, d, y, p1)
            return term
        case _:
            return term


def reduce_inside_step(term: Term, red: Callable[[Term], Term]) -> Term:

    t1 = red(term)
    if t1 != term:
        return t1

    reducer = lambda term: reduce_inside_step(term, red)

    match term:
        case App(f, a):
            f1 = reducer(f)
            if f1 != f:
                return App(f1, a)
            a1 = reducer(a)
            if a1 != a:
                return App(f, a1)
            return term

        case Lam(ty, body):
            ty1 = reducer(ty)
            if ty1 != ty:
                return Lam(ty1, body)
            body1 = reducer(body)
            if body1 != body:
                return Lam(ty, body1)
            return term

        case Pi(ty, body):
            ty1 = reducer(ty)
            if ty1 != ty:
                return Pi(ty1, body)
            body1 = reducer(body)
            if body1 != body:
                return Pi(ty, body1)
            return term

        case Id(ty, l, r):
            ty1 = reducer(ty)
            if ty1 != ty:
                return Id(ty1, l, r)
            l1 = reducer(l)
            if l1 != l:
                return Id(ty, l1, r)
            r1 = reducer(r)
            if r1 != r:
                return Id(ty, l, r1)
            return term

        case Refl(ty, t0):
            ty1 = reducer(ty)
            if ty1 != ty:
                return Refl(ty1, t0)
            t1 = reducer(t0)
            if t1 != t0:
                return Refl(ty, t1)
            return term

        case IdElim(A, x, P, d, y, p):
            A1 = reducer(A)
            if A1 != A:
                return IdElim(A1, x, P, d, y, p)
            x1 = reducer(x)
            if x1 != x:
                return IdElim(A, x1, P, d, y, p)
            P1 = reducer(P)
            if P1 != P:
                return IdElim(A, x, P1, d, y, p)
            d1 = reducer(d)
            if d1 != d:
                return IdElim(A, x, P, d1, y, p)
            y1 = reducer(y)
            if y1 != y:
                return IdElim(A, x, P, d, y1, p)
            p1 = reducer(p)
            if p1 != p:
                return IdElim(A, x, P, d, y, p1)
            return term

        case InductiveElim(inductive, motive, cases, scrutinee):
            motive1 = reducer(motive)
            if motive1 != motive:
                return InductiveElim(inductive, motive1, cases, scrutinee)
            cases1 = [reducer(branch) for branch in cases]
            if cases1 != cases:
                return InductiveElim(inductive, motive, cases1, scrutinee)
            scrutinee1 = reducer(scrutinee)
            if scrutinee1 != scrutinee:
                return InductiveElim(inductive, motive, cases, scrutinee1)
            return term

        case Var() | Univ() | InductiveType() | InductiveConstructor():
            return term

    raise TypeError(f"Unexpected term in reducer: {term!r}")

__all__ = ["whnf", "reduce_inside_step"]
