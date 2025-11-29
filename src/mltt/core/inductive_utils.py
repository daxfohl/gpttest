"""Shared helpers for inductive types."""

from __future__ import annotations

from .ast import App, InductiveConstructor, InductiveType, Term
from .debruijn import subst


def apply_term(term: Term, args: tuple[Term, ...]) -> Term:
    """Apply ``args`` to ``term`` left-associatively."""
    result: Term = term
    for arg in args:
        result = App(result, arg)
    return result


def decompose_app(term: Term) -> tuple[Term, tuple[Term, ...]]:
    """Split an application into its head and argument tuple."""
    args: list[Term] = []
    head = term
    while isinstance(head, App):
        args.insert(0, head.arg)
        head = head.func
    return head, tuple(args)


def decompose_ctor_app(
    term: Term,
) -> tuple[InductiveConstructor, tuple[Term, ...]] | None:
    """Return the constructor head and arguments if ``term`` is applied."""
    head, args = decompose_app(term)
    if isinstance(head, InductiveConstructor):
        return head, args
    return None


def instantiate_params_indices(
    term: Term,
    params: tuple[Term, ...],
    indices: tuple[Term, ...],
    offset: int = 0,
) -> Term:
    """Substitute ``params``/``indices`` (params outermost, indices next)."""
    result = term
    for idx, param in enumerate(params):
        j = offset + len(indices) + (len(params) - 1 - idx)
        result = subst(result, param, j=j)
    for idx, index in enumerate(indices):
        j = offset + (len(indices) - 1 - idx)
        result = subst(result, index, j=j)
    return result


def match_inductive_application(
    term: Term, inductive: InductiveType
) -> tuple[tuple[Term, ...], tuple[Term, ...]] | None:
    """Return param/index args when ``term`` is an applied ``inductive``."""
    head, args = decompose_app(term)
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    total = param_count + index_count
    if head is inductive and len(args) == total:
        return args[:param_count], args[param_count:]
    return None


def ctor_index(inductive: InductiveType, ctor: InductiveConstructor) -> int:
    """Position of ``ctor`` inside ``inductive.constructors``."""
    for idx, ctor_def in enumerate(inductive.constructors):
        if ctor is ctor_def:
            return idx
    raise TypeError("Constructor does not belong to inductive type")


__all__ = [
    "apply_term",
    "ctor_index",
    "decompose_app",
    "decompose_ctor_app",
    "instantiate_params_indices",
    "match_inductive_application",
]
