"""Shared helpers for inductive types."""

from __future__ import annotations

from .ast import App, Ctor, I, Term
from .debruijn import subst


def apply_term(term: Term, *args: Term) -> Term:
    """Apply ``args`` to ``term`` left-associatively.

    Constructors and inductive type heads are stored unapplied; callers often
    need to thread parameters, indices, and payloads in order. This helper
    keeps those call sites readable and centralizes the left-associative
    application pattern.
    """
    #  e.g. term = \x->(\y->z). args = [x, y]
    result: Term = term
    for arg in args:
        result = App(result, arg)
    return result


def decompose_app(term: Term) -> tuple[Term, tuple[Term, ...]]:
    """Split an application into its head and argument tuple.

    This is the inverse of ``apply_term`` and is used by eliminator matching.
    It peels applications from the outside in, yielding the ultimate head
    (which may itself be an inductive type or constructor) and the ordered
    argument tuple.
    """
    #  e.g. input = ((((\x->(\y->z)) x) y). output: [\x->(\y->z), [x, y]]
    args: list[Term] = []
    while isinstance(term, App):
        args.append(term.arg)
        term = term.func
    return term, tuple(reversed(args))


def decompose_ctor_app(
    term: Term,
) -> tuple[Ctor, tuple[Term, ...]] | None:
    """Return the constructor head and arguments if ``term`` is applied.

    Returns ``None`` when the head is not a constructor or the term is not an
    application chain.
    """
    head, args = decompose_app(term)
    if isinstance(head, Ctor):
        return head, args
    # For example, it could be a Var, or an axiom like LEM.
    return None


def instantiate_forward(
    schema_tys: tuple[Term, ...],
    actual_args: tuple[Term, ...],
) -> tuple[Term, ...]:
    out: list[Term] = []
    for i, schema in enumerate(schema_tys):
        inst = schema
        for j in range(i):
            inst = subst(inst, actual_args[j], i - j - 1)
        out.append(inst)
    return tuple(out)


def instantiate_into(
    params: tuple[Term, ...], target: tuple[Term, ...]
) -> tuple[Term, ...]:
    output = []
    for i, arg in enumerate(target):
        for j, param in enumerate(params):
            index = len(params) + i - j - 1
            arg = subst(arg, param, index)
        output.append(arg)
    return tuple(output)


def instantiate_params_indices(
    term: Term,
    params: tuple[Term, ...],
    indices: tuple[Term, ...],
    offset: int = 0,
) -> Term:
    """Substitute ``params``/``indices`` (params outermost, indices next).

    Parameters live outermost, followed by indices; both are ordered from
    outer to inner. ``offset`` lets callers skip over constructor arguments
    already in scope. Substitutions run from outermost to innermost so De
    Bruijn shifts line up.
    """
    result = term
    for idx, param in enumerate(params):
        j = offset + len(indices) + (len(params) - 1 - idx)
        # j=0+2+2-1-0=3
        # j=0+2+2-1-1=2
        result = subst(result, param, j=j)
    for idx, index in enumerate(indices):
        # j=0+2-1-0=1
        # j=0+2-1-1=0
        j = offset + (len(indices) - 1 - idx)
        result = subst(result, index, j=j)
    return result


def match_inductive_application(
    term: Term, inductive: I
) -> tuple[tuple[Term, ...], tuple[Term, ...]] | None:
    """Return param/index args when ``term`` is an applied ``inductive``.

    Matches only fully-applied occurrences (same param/index arity).
    """
    head, args = decompose_app(term)
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    total = param_count + index_count
    if head is inductive and len(args) == total:
        return args[:param_count], args[param_count:]
    return None


def ctor_index(ctor: Ctor) -> int:
    """Position of ``ctor`` inside ``inductive.constructors``."""
    for idx, ctor_def in enumerate(ctor.inductive.constructors):
        if ctor is ctor_def:
            return idx
    raise TypeError("Constructor does not belong to inductive type")


__all__ = [
    "apply_term",
    "ctor_index",
    "decompose_app",
    "decompose_ctor_app",
    "instantiate_into",
    "instantiate_forward",
    "match_inductive_application",
]
