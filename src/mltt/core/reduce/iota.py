"""Iota reduction helpers for inductive eliminators and identity types."""

from __future__ import annotations

from ..ast import (
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
from ..debruijn import subst


def _apply_term(term: Term, args: tuple[Term, ...]) -> Term:
    result: Term = term
    for arg in args:
        result = App(result, arg)
    return result


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
    params: tuple[Term, ...],
    indices: tuple[Term, ...],
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
    args: tuple[Term, ...],
) -> Term:
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    index = _ctor_index(inductive, ctor)
    if index >= len(cases):
        return InductiveElim(inductive, motive, cases, _apply_term(ctor, args))
    branch = cases[index]

    if len(args) < param_count + index_count:
        return InductiveElim(inductive, motive, cases, _apply_term(ctor, args))
    param_args = args[:param_count]
    index_args = args[param_count : param_count + index_count]
    ctor_args = args[param_count + index_count :]
    if len(ctor_args) != len(ctor.arg_types):
        return InductiveElim(inductive, motive, cases, _apply_term(ctor, args))

    index_arg_types = [
        _instantiate_params_indices(index_ty, param_args, (), offset=0)
        for index_ty in inductive.index_types
    ]

    instantiated_arg_types = [
        _instantiate_params_indices(arg_ty, param_args, index_args, offset=idx)
        for idx, arg_ty in enumerate(ctor.arg_types)
    ]

    recursive_counts = sum(
        1
        for arg_ty in instantiated_arg_types
        if (match := _match_inductive_application(arg_ty, inductive))
        and len(match[0]) == len(param_args)
        and all(param == arg_param for param, arg_param in zip(match[0], param_args))
    )
    binder_count = len(instantiated_arg_types) + recursive_counts

    applied_args: list[Term] = []
    branch = cases[index]

    lam_count = 0
    branch_scan = branch
    while isinstance(branch_scan, Lam):
        lam_count += 1
        branch_scan = branch_scan.body

    extra_needed = max(0, lam_count - binder_count)
    branch_for_indices = branch
    for idx_arg, idx_ty in zip(index_args, index_arg_types):
        if extra_needed <= 0:
            break
        if isinstance(branch_for_indices, Lam) and branch_for_indices.ty == idx_ty:
            applied_args.append(idx_arg)
            branch_for_indices = branch_for_indices.body
            extra_needed -= 1
        else:
            break

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


__all__ = ["iota_head_step", "iota_step"]
