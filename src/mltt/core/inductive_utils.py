"""Shared helpers for inductive types."""

from __future__ import annotations

from typing import Sequence, TypeVar

from .ast import Ctor, Term
from .debruijn import subst, shift

T = TypeVar("T")


def instantiate_ctor_arg_types(
    ctor_arg_types: Sequence[Term],
    params_actual: Sequence[Term],
) -> tuple[Term, ...]:
    schemas: list[Term] = []
    p = len(params_actual)
    for i, schema in enumerate(ctor_arg_types):
        t = schema
        # eliminate param binders outermost → innermost at their indexed depth
        for s, param in enumerate(params_actual):
            index = i + p - s - 1
            t = subst(t, shift(param, i + (p - s - 1)), index)
        schemas.append(t)
    return tuple(schemas)


def instantiate_ctor_result_indices_under_fields(
    result_indices: tuple[Term, ...],
    params_actual_shifted: tuple[Term, ...],  # already shifted by m at callsite
    m: int,  # number of ctor fields in scope
) -> tuple[Term, ...]:
    """
    result_indices schemas are written in context (params)(fields).
    We are currently in context Γ,(fields) (params already in Γ but shifted by m).
    Discharge params binders only; keep field vars (0..m-1) intact.
    """
    p = len(params_actual_shifted)
    out: list[Term] = []
    for schema in result_indices:
        t = schema
        # eliminate param binders outermost → innermost so inner indices stay stable
        for s, param in enumerate(params_actual_shifted):
            index = m + p - s - 1
            t = subst(t, shift(param, p - s - 1), index)
        out.append(t)
    return tuple(out)


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
    "ctor_index",
    "instantiate_ctor_arg_types",
    "instantiate_ctor_result_indices_under_fields",
]
