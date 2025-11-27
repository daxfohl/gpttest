"""Beta reduction and evaluation helpers for MLTT terms."""

from __future__ import annotations

from typing import Sequence

from .ast import (
    App,
    InductiveConstructor,
    InductiveElim,
    InductiveType,
    Lam,
    Pi,
    Term,
    Univ,
    Var,
)
from .debruijn import subst


def _apply_term(term: Term, args: Sequence[Term]) -> Term:
    result: Term = term
    for arg in args:
        result = App(result, arg)
    return result


def _apply_ctor(ctor: InductiveConstructor, args: Sequence[Term]) -> Term:
    return _apply_term(ctor, args)


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


def _decompose_app(term: Term) -> tuple[Term, tuple[Term, ...]]:
    args: list[Term] = []
    head = term
    while isinstance(head, App):
        args.insert(0, head.arg)
        head = head.func
    return head, tuple(args)


def _instantiate_params_indices(
    term: Term,
    params: Sequence[Term],
    indices: Sequence[Term],
    offset: int = 0,
) -> Term:
    result = term
    for idx, param in enumerate(reversed(params)):
        result = subst(result, param, j=offset + len(indices) + idx)
    for idx, index in enumerate(reversed(indices)):
        result = subst(result, index, j=offset + idx)
    return result


def _match_inductive_application(
    term: Term, inductive: InductiveType
) -> tuple[tuple[Term, ...], tuple[Term, ...]] | None:
    head, args = _decompose_app(term)
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    total = param_count + index_count
    if head is inductive and len(args) == total:
        return args[:param_count], args[param_count:]
    return None


def _ctor_index(inductive: InductiveType, ctor: InductiveConstructor) -> int:
    for idx, ctor_def in enumerate(inductive.constructors):
        if ctor is ctor_def:
            return idx
    raise TypeError("Constructor does not belong to inductive type")


def _iota_constructor(
    inductive: InductiveType,
    motive: Term,
    cases: list[Term],
    ctor: InductiveConstructor,
    args: Sequence[Term],
) -> Term:
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    index = _ctor_index(inductive, ctor)
    if index >= len(cases):
        return InductiveElim(inductive, motive, cases, _apply_ctor(ctor, args))
    branch = cases[index]

    if len(args) < param_count + index_count:
        return InductiveElim(inductive, motive, cases, _apply_ctor(ctor, args))
    param_args = args[:param_count]
    index_args = args[param_count : param_count + index_count]
    ctor_args = args[param_count + index_count :]
    if len(ctor_args) != len(ctor.arg_types):
        return InductiveElim(inductive, motive, cases, _apply_ctor(ctor, args))

    instantiated_arg_types = [
        _instantiate_params_indices(arg_ty, param_args, index_args, offset=idx)
        for idx, arg_ty in enumerate(ctor.arg_types)
    ]
    applied_args: list[Term] = []
    for arg_ty, arg in zip(instantiated_arg_types, ctor_args, strict=False):
        applied_args.append(arg)
        match _match_inductive_application(arg_ty, inductive):
            case (ctor_params, _):
                if len(ctor_params) == len(param_args) and all(
                    param == arg_param
                    for param, arg_param in zip(ctor_params, param_args)
                ):
                    applied_args.append(InductiveElim(inductive, motive, cases, arg))
            case _:
                pass

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

        case InductiveElim(inductive, motive, cases, scrutinee):
            motive1 = beta_step(motive)
            if motive1 != motive:
                return InductiveElim(inductive, motive1, cases, scrutinee)
            cases1 = [beta_step(branch) for branch in cases]
            if cases1 != cases:
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
                expected_args = (
                    len(inductive.param_types)
                    + len(inductive.index_types)
                    + len(ctor.arg_types)
                )
                if ctor.inductive is inductive and len(args) == expected_args:
                    return _iota_constructor(inductive, motive, cases, ctor, args)
            scrutinee1 = iota_head_step(scrutinee)
            if scrutinee1 != scrutinee:
                return InductiveElim(inductive, motive, cases, scrutinee1)
            return t
        case _:
            return t


def iota_step(term: Term) -> Term:
    """One iota-reduction step anywhere (InductiveElim)."""

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

        case InductiveElim(inductive, motive, cases, scrutinee):
            motive1 = iota_step(motive)
            if motive1 != motive:
                return InductiveElim(inductive, motive1, cases, scrutinee)
            cases1 = [iota_step(branch) for branch in cases]
            if cases1 != cases:
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
