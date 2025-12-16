"""Iota reduction helpers for inductive eliminators and identity types."""

from __future__ import annotations

from ..ast import App, Ctor, Elim, I, Lam, Pi, Term, Univ, Var
from ..inductive_utils import (
    apply_term,
    ctor_index,
    decompose_ctor_app,
    decompose_app,
    instantiate_ctor_arg_types,
)


def _iota_constructor(
    inductive: I,
    motive: Term,
    cases: tuple[Term, ...],
    ctor: Ctor,
    args: tuple[Term, ...],
) -> Term:
    """Compute the iota-reduction of an eliminator on a fully-applied ctor."""
    p = len(inductive.param_types)

    params_actual = args[:p]
    ctor_args = args[p:]  # constructor fields (length m)

    case = cases[ctor_index(ctor)]

    # Instantiate ctor field types so we can detect which fields are recursive.
    inst_arg_types = instantiate_ctor_arg_types(ctor.arg_types, params_actual)

    # Build IHs in the same order you used when you built ih_types for the telescope.
    ihs: list[Term] = []
    for field_term, field_ty in zip(ctor_args, inst_arg_types, strict=True):
        head, _ = decompose_app(field_ty)
        if head is inductive:
            ihs.append(
                Elim(
                    inductive=inductive,
                    motive=motive,
                    cases=cases,
                    scrutinee=field_term,
                )
            )

    # Iota result: apply case to ctor fields and IHs (and nothing else),
    # because that is exactly how you type-checked the case telescope.
    return apply_term(case, *ctor_args, *ihs)


def iota_head_step(t: Term) -> Term:
    """Perform one iota step at the head position, if possible."""
    match t:
        case Elim(inductive, motive, cases, scrutinee):
            # Try to reduce when the scrutinee is a fully-applied constructor.
            decomposition = decompose_ctor_app(scrutinee)
            if decomposition:
                ctor, args = decomposition
                expected_args = len(inductive.param_types) + len(ctor.arg_types)
                if ctor.inductive is inductive and len(args) == expected_args:
                    return _iota_constructor(inductive, motive, cases, ctor, args)
                raise ValueError()
            # Otherwise, attempt to reduce inside the scrutinee.
            scrutinee1 = iota_head_step(scrutinee)
            if scrutinee1 != scrutinee:
                return Elim(inductive, motive, cases, scrutinee1)
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
