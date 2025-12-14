"""Shared helpers for inductive types."""

from __future__ import annotations

from itertools import islice
from typing import Sequence, Any, TypeVar, Iterator

from .ast import App, Ctor, I, Term, Lam, Pi, Var
from .debruijn import shift, subst

T = TypeVar("T")


def apply_term(term: Term, *args: Term) -> Term:
    """Apply ``args`` to ``term`` left-associatively.

    Constructors and inductive type heads are stored unapplied; callers often
    need to thread parameters, indices, and payloads in order. This helper
    keeps those call sites readable and centralizes the left-associative
    application pattern.

    Args:
        term: Function being applied.
        *args: Arguments to apply, ordered left-to-right.

    Returns:
        The left-associated application ``(((term arg0) arg1) ...)``.
    """
    #  e.g. term = \x->(\y->z). args = [x, y]
    result: Term = term
    for arg in args:
        result = App(result, arg)
    return result


def nested_lam(*param_tys: Term, body: Term) -> Term:
    """Build a right-nested lambda chain over ``param_tys`` ending in ``body``.

    Each element of ``param_tys`` becomes one binder, with the first argument
    binding outermost and the last argument closest to ``body``. This mirrors
    the left-to-right order of parameters in source syntax while keeping the
    resulting AST compact and easy to construct programmatically.

    Args:
        *param_tys: Parameter types, ordered from outermost to innermost.
        body: The lambda body that closes over the introduced binders.

    Returns:
        A ``Lam`` chain whose body is ``body`` and whose binders match
        ``param_tys`` in order.
    """
    fn: Term = body
    for param_ty in reversed(param_tys):
        fn = Lam(param_ty, fn)
    return fn


def nested_pi(*param_tys: Term, return_ty: Term) -> Term:
    """Build a right-nested Pi chain over ``param_tys`` ending in ``return_ty``.

    Like ``nested_lam``, the outermost quantifier corresponds to the first
    element of ``param_tys`` while the last element binds closest to
    ``return_ty``. This helper centralizes the repetitive Pi-tower pattern
    used throughout the inductive definitions.

    Args:
        *param_tys: Parameter types, ordered from outermost to innermost.
        return_ty: The codomain of the innermost Pi.

    Returns:
        A ``Pi`` chain whose codomain is ``return_ty`` and whose binders match
        ``param_tys`` in order.
    """
    pi: Term = return_ty
    for param_ty in reversed(param_tys):
        pi = Pi(param_ty, pi)
    return pi


def decompose_app(term: Term) -> tuple[Term, tuple[Term, ...]]:
    """Split an application into its head and argument tuple.

    This is the inverse of ``apply_term`` and is used by eliminator matching.
    It peels applications from the outside in, yielding the ultimate head
    (which may itself be an inductive type or constructor) and the ordered
    argument tuple.

    Args:
        term: Term to break apart. Non-application terms return themselves as
            the head with an empty argument tuple.

    Returns:
        A pair ``(head, args)`` where ``head`` is the unapplied function and
        ``args`` are in the same order they were originally applied.
    """
    #  e.g. input = ((((\x->(\y->z)) x) y). output: [\x->(\y->z), [x, y]]
    args: list[Term] = []
    while isinstance(term, App):
        args.append(term.arg)
        term = term.func
    return term, tuple(reversed(args))


def decompose_lam(term: Term) -> tuple[Term, tuple[Term, ...]]:
    args: list[Term] = []
    while isinstance(term, Lam):
        args.append(term.arg_ty)
        term = term.body
    return term, tuple(args)


def decompose_pi(term: Term) -> tuple[Term, tuple[Term, ...]]:
    args: list[Term] = []
    while isinstance(term, Pi):
        args.append(term.arg_ty)
        term = term.return_ty
    return term, tuple(args)


def decompose_ctor_app(
    term: Term,
) -> tuple[Ctor, tuple[Term, ...]] | None:
    """Return the constructor head and arguments if ``term`` is applied.

    Args:
        term: Candidate constructor application.

    Returns:
        ``(Ctor, args)`` when the head is a constructor, otherwise ``None``.
    """
    head, args = decompose_app(term)
    if isinstance(head, Ctor):
        return head, args
    # For example, it could be a Var, or an axiom like LEM.
    return None


def instantiate_into(
    params: tuple[Term, ...], target: tuple[Term, ...]
) -> tuple[Term, ...]:
    """Instantiate ``target`` types with ``params`` inserted outermost-first.

    Args:
        params: Parameter terms to substitute, ordered from outermost to
            innermost.
        target: Types whose de Bruijn references are adjusted as parameters are
            threaded in.

    Returns:
        A tuple of instantiated target types.
    """
    output = []
    for i, arg in enumerate(target):
        for j, param in enumerate(params):
            index = i + len(params) - j - 1
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

    Args:
        term: Target term whose variables are instantiated.
        params: Parameter terms, ordered outermost to innermost.
        indices: Index terms, ordered outermost to innermost.
        offset: Number of binders already in scope (e.g., constructor args).

    Returns:
        The instantiated term.
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


def instantiate_for_inductive(
    inductive: I,
    params: tuple[Term, ...],
    indices: tuple[Term, ...],
    targets: tuple[Term, ...],
    args: tuple[Term, ...] = (),
) -> tuple[Term, ...]:
    """Instantiate ``targets`` using the inductive param/index ordering.

    Parameters are outermost, followed by indices, then optional constructor
    arguments supplied via ``args``. Params/indices are shifted by the inductive
    index arity so they remain stable when new binders are introduced.
    """

    shifted = tuple(
        shift(arg, len(inductive.index_types)) for arg in (*params, *indices)
    )
    return instantiate_into((*shifted, *args), targets)


def split_to_match(
    seq: Sequence[T], *shape: Sequence[Any]
) -> tuple[tuple[T, ...], ...]:
    """
    Splits a sequence into segments to match the lengths (shape)
    of an existing sequence of sequences.
    """
    seq_iter: Iterator[T] = iter(seq)
    return tuple(tuple(islice(seq_iter, len(sublist))) for sublist in shape)


def match_inductive_application(
    term: Term, inductive: I
) -> tuple[tuple[Term, ...], tuple[Term, ...]] | None:
    """Return param/index args when ``term`` is an applied ``inductive``.

    Matches only fully-applied occurrences (same param/index arity).

    Args:
        term: Candidate inductive application.
        inductive: Inductive head to match against.

    Returns:
        ``(params, indices)`` when fully applied, otherwise ``None``.
    """
    head, args = decompose_app(term)
    param_count = len(inductive.param_types)
    index_count = len(inductive.index_types)
    total = param_count + index_count
    if head is inductive and len(args) == total:
        return args[:param_count], args[param_count:]
    return None


def ctor_index(ctor: Ctor) -> int:
    """Position of ``ctor`` inside ``inductive.constructors``.

    Args:
        ctor: Constructor whose position is requested.

    Returns:
        Zero-based index of the constructor.

    Raises:
        TypeError: If ``ctor`` is not part of its ``inductive``.
    """
    for idx, ctor_def in enumerate(ctor.inductive.constructors):
        if ctor is ctor_def:
            return idx
    raise TypeError("Constructor does not belong to inductive type")


__all__ = [
    "apply_term",
    "ctor_index",
    "decompose_app",
    "decompose_ctor_app",
    "decompose_lam",
    "decompose_pi",
    "instantiate_into",
    "match_inductive_application",
    "nested_lam",
    "nested_pi",
    "split_to_match",
]
