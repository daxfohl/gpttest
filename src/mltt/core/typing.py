"""Thin wrappers around ``Term`` typing operations."""

from __future__ import annotations

from .ast import Term
from .debruijn import Ctx


def expect_universe(term: Term, ctx: Ctx | None = None) -> int:
    return term.expect_universe(ctx)


def type_equal(t1: Term, t2: Term, ctx: Ctx | None = None) -> bool:
    return t1.type_equal(t2, ctx)


def infer_type(term: Term, ctx: Ctx | None = None) -> Term:
    return term.infer_type(ctx)


def type_check(term: Term, ty: Term, ctx: Ctx | None = None) -> None:
    term.type_check(ty, ctx)


__all__ = ["type_equal", "infer_type", "type_check", "expect_universe"]
