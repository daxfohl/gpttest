"""Metavariable entries managed by the solver."""

from __future__ import annotations

from dataclasses import dataclass

from mltt.common.span import Span
from mltt.kernel.ast import Term


@dataclass
class Meta:
    ctx_len: int
    ty: Term
    solution: Term | None = None
    span: Span | None = None
    kind: str = "hole"
