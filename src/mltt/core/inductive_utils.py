"""Shared helpers for inductive types."""

from __future__ import annotations

from itertools import islice
from typing import Sequence, Any, TypeVar, Iterator

from .ast import App, Ctor, Term, Lam, Pi
from .debruijn import subst, shift

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


def instantiate_ctor_arg_types(
    ctor_arg_types: tuple[Term, ...],
    params_actual: tuple[Term, ...],
) -> tuple[Term, ...]:
    p = len(params_actual)
    out: list[Term] = []

    for i, schema in enumerate(ctor_arg_types):
        t = schema

        # substitute params (outermost to innermost) at descending indices
        # param0 is farthest: index = i + (p-1)
        for s in range(p):
            j = i + (p - 1 - s)
            t = subst(t, shift(params_actual[s], i), j)

        out.append(t)

    return tuple(out)


def instantiate_ctor_result_indices(
    result_indices: tuple[Term, ...],
    params_actual: tuple[Term, ...],
    ctor_args: tuple[Term, ...],
) -> tuple[Term, ...]:
    """
    Instantiate ctor.result_indices schemas.

    The schema context is (params)(fields). We substitute the actual params and
    constructor fields (in that order) to obtain indices in the ambient
    context, with all binders discharged.
    """
    p = len(params_actual)
    m = len(ctor_args)

    out: list[Term] = []
    for schema in result_indices:
        t = schema

        # 1) Substitute params. They sit outermost, above all ctor args.
        for s in range(p):
            j = m + (p - 1 - s)  # from m+p-1 down to m
            t = subst(t, shift(params_actual[s], m), j)

        # 2) Substitute constructor fields from outermost to innermost.
        for s in range(m):
            j = m - 1 - s  # from m-1 down to 0
            t = subst(t, ctor_args[s], j)

        out.append(t)

    return tuple(out)


# def instantiate_ctor_result_indices(
#     result_indices: tuple[Term, ...],
#     params_actual: tuple[Term, ...],
#     indices_actual: tuple[Term, ...],
#     m: int,  # number of ctor args
# ) -> tuple[Term, ...]:
#     p = len(params_actual)
#     q = len(indices_actual)
#
#     out: list[Term] = []
#     for schema in result_indices:
#         t = schema
#
#         for s in range(p):
#             j = m + q + (p - 1 - s)
#             t = subst(t, shift(params_actual[s], m), j)
#
#         for s in range(q):
#             j = m + (q - 1 - s)
#             t = subst(t, shift(indices_actual[s], m), j)
#
#         out.append(t)
#
#     return tuple(out)


def instantiate_into(*params: Term, target: tuple[Term, ...]) -> tuple[Term, ...]:
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


def split_to_match(
    seq: Sequence[T], *shape: Sequence[Any]
) -> tuple[tuple[T, ...], ...]:
    """
    Splits a sequence into segments to match the lengths (shape)
    of an existing sequence of sequences.
    """
    seq_iter: Iterator[T] = iter(seq)
    return tuple(tuple(islice(seq_iter, len(sublist))) for sublist in shape)


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
    "nested_lam",
    "nested_pi",
    "split_to_match",
    "instantiate_ctor_arg_types",
    "instantiate_ctor_result_indices",
]
