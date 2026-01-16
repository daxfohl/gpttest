"""Argument matcher for surface applications (positional/implicit/named)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mltt.common.span import Span
from mltt.elab.ast import EArg, ENamedArg
from mltt.elab.errors import ElabError
from mltt.elab.types import BinderSpec


@dataclass(frozen=True)
class ArgDecision:
    kind: Literal["explicit", "implicit", "missing", "stop"]
    arg: EArg | ENamedArg | None = None


class ArgMatcher:
    """Match surface arguments against binder specs."""

    def __init__(
        self,
        binders: tuple[BinderSpec, ...],
        args: tuple[EArg, ...],
        named_args: tuple[ENamedArg, ...],
        span: Span,
    ) -> None:
        self._binders = binders
        self._args = list(args)
        self._named: dict[str, ENamedArg] = {arg.name: arg for arg in named_args}
        self._used_named: set[str] = set()
        self._pos = 0
        self._span = span
        if named_args and any(b.name is None for b in binders):
            raise ElabError("Named arguments require binder names", span)

    def has_positional(self) -> bool:
        return self._pos < len(self._args)

    def unknown_named(self) -> str | None:
        for name in self._named:
            if name not in self._used_named:
                return name
        return None

    def match_for_binder(self, binder: BinderSpec, allow_partial: bool) -> ArgDecision:
        if binder.implicit:
            return self._match_implicit(binder, allow_partial)
        return self._match_explicit(binder, allow_partial)

    def _match_implicit(self, binder: BinderSpec, allow_partial: bool) -> ArgDecision:
        if self.has_positional():
            arg = self._args[self._pos]
            if arg.implicit:
                self._pos += 1
                return ArgDecision(kind="explicit", arg=arg)
            return ArgDecision(kind="implicit")
        if binder.name is not None and binder.name in self._named:
            self._used_named.add(binder.name)
            return ArgDecision(kind="explicit", arg=self._named[binder.name])
        return ArgDecision(kind="implicit")

    def _match_explicit(self, binder: BinderSpec, allow_partial: bool) -> ArgDecision:
        if self.has_positional():
            arg = self._args[self._pos]
            if arg.implicit:
                raise ElabError(
                    "Implicit argument provided where explicit expected", self._span
                )
            self._pos += 1
            return ArgDecision(kind="explicit", arg=arg)
        if binder.name is not None and binder.name in self._named:
            self._used_named.add(binder.name)
            return ArgDecision(kind="explicit", arg=self._named[binder.name])
        if allow_partial:
            if self.unknown_named() is not None:
                return ArgDecision(kind="missing")
            return ArgDecision(kind="stop")
        return ArgDecision(kind="missing")
