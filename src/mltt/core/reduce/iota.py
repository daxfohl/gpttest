"""Iota reduction helpers for inductive eliminators and identity types."""

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
from ..inductive_utils import (
    apply_term,
    ctor_index,
    decompose_ctor_app,
    decompose_app,
)


def _iota_constructor(
    inductive: I,
    motive: Term,
    cases: tuple[Term, ...],
    ctor: Ctor,
    args: tuple[Term, ...],
) -> Term:
    """Compute the iota-reduction of an eliminator on a fully-applied ctor."""
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    index = ctor_index(ctor)
    case = cases[index]
    ctor_args = args[param_count + index_count :]

    ihs: list[Term] = []
    for arg_term, arg_ty in zip(ctor_args, ctor.arg_types, strict=True):
        head, head_args = decompose_app(arg_ty)
        if head is ctor.inductive:
            # only works if after substituting param_args and index_args into ctor_arg_types.
            # assert head_args[:param_count] == param_args, f"{arg_ty}: {head_args[:param_count]!r} == {param_args}"
            ih = Elim(
                inductive=ctor.inductive,
                motive=motive,
                cases=cases,
                scrutinee=arg_term,
            )
            ihs.append(ih)

    all_args = (*ctor_args, *ihs)
    test = apply_term(case, all_args)
    return test


def iota_head_step(t: Term) -> Term:
    """Perform one iota step at the head position, if possible."""
    match t:
        case Elim(inductive, motive, cases, scrutinee):
            # Try to reduce when the scrutinee is a fully-applied constructor.
            decomposition = decompose_ctor_app(scrutinee)
            if decomposition:
                ctor, args = decomposition
                expected_args = (
                    len(inductive.param_types)
                    + len(inductive.index_types)
                    + len(ctor.arg_types)
                )
                if ctor.inductive is inductive and len(args) == expected_args:
                    return _iota_constructor(inductive, motive, cases, ctor, args)
                raise ValueError()
            # Otherwise, attempt to reduce inside the scrutinee.
            scrutinee1 = iota_head_step(scrutinee)
            if scrutinee1 != scrutinee:
                return Elim(inductive, motive, cases, scrutinee1)
            return t
        case IdElim(A, x, P, d, y, Refl(_, _)):
            return d
        case IdElim(A, x, P, d, y, p):
            # Push reduction into the proof if the head does not expose a Refl.
            p1 = iota_head_step(p)
            if p1 != p:
                return IdElim(A, x, P, d, y, p1)
            return t
        case _:
            return t


def iota_step(term: Term) -> Term:
    """One iota-reduction step anywhere (InductiveElim / IdElim).

    Tries a head step first; if that fails, walks the term recursively to find
    a reducible subterm.
    """

    t1 = iota_head_step(term)
    if t1 != term:
        return t1

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

        case Elim(inductive, motive, cases, scrutinee):
            motive1 = iota_step(motive)
            if motive1 != motive:
                return Elim(inductive, motive1, cases, scrutinee)
            cases1 = tuple(iota_step(case) for case in cases)
            if cases1 != cases:
                return Elim(inductive, motive, cases1, scrutinee)
            scrutinee1 = iota_step(scrutinee)
            if scrutinee1 != scrutinee:
                return Elim(inductive, motive, cases, scrutinee1)
            return term

        case Var(_) | Univ() | I() | Ctor():
            return term

    raise TypeError(f"Unexpected term in iota_step: {term!r}")


__all__ = ["iota_head_step", "iota_step"]
