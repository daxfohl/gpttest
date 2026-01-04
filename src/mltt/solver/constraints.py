"""Constraint data structures for the solver."""

from __future__ import annotations

from dataclasses import dataclass

from mltt.common.span import Span
from mltt.kernel.ast import Term


@dataclass
class Constraint:
    ctx_len: int
    lhs: Term
    rhs: Term
    span: Span | None = None
    kind: str = "term_eq"
