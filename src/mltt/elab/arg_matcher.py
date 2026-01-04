"""Argument matching helpers for applications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from mltt.elab.east import EArg
from mltt.elab.etype import ElabBinderInfo
from mltt.surface.sast import Span, SurfaceError


ArgKind = Literal["explicit", "implicit", "missing", "stop"]


@dataclass(frozen=True)
class ArgDecision:
    kind: ArgKind
    arg: EArg | None = None
    from_named: bool = False


class ArgMatcher:
    def __init__(
        self,
        binders: tuple[ElabBinderInfo, ...],
        args: tuple[EArg, ...],
        span: Span,
    ) -> None:
        self._span = span
        named_seen = False
        for item in args:
            if item.name is not None:
                named_seen = True
            elif named_seen:
                raise SurfaceError(
                    "Positional arguments must come before named arguments",
                    item.term.span,
                )
        self._positional = [arg for arg in args if arg.name is None]
        named: dict[str, EArg] = {}
        for item in args:
            if item.name is None:
                continue
            if item.name in named:
                raise SurfaceError(
                    f"Duplicate named argument {item.name}", item.term.span
                )
            named[item.name] = item
        if named and not any(b.name for b in binders):
            raise SurfaceError("Named arguments require binder names", span)
        self._named = named

    def match_for_binder(
        self, binder: ElabBinderInfo, *, allow_partial: bool
    ) -> ArgDecision:
        binder_name = binder.name
        if binder_name is not None and binder_name in self._named:
            return ArgDecision(
                kind="explicit",
                arg=self._named.pop(binder_name),
                from_named=True,
            )
        if self._positional:
            candidate = self._positional[0]
            if binder.implicit:
                if candidate.implicit:
                    self._positional.pop(0)
                    return ArgDecision(kind="explicit", arg=candidate)
                return ArgDecision(kind="implicit")
            if candidate.implicit:
                raise SurfaceError(
                    "Implicit argument provided where explicit expected",
                    candidate.term.span,
                )
            self._positional.pop(0)
            return ArgDecision(kind="explicit", arg=candidate)
        if binder.implicit:
            return ArgDecision(kind="implicit")
        if allow_partial and not self._named:
            return ArgDecision(kind="stop")
        if allow_partial:
            return ArgDecision(kind="missing")
        raise SurfaceError("Missing explicit argument", self._span)

    def has_positional(self) -> bool:
        return bool(self._positional)

    def has_named(self) -> bool:
        return bool(self._named)

    def unknown_named(self) -> str | None:
        if not self._named:
            return None
        return next(iter(self._named.keys()))

    def next_arg_span(self) -> Span:
        if self._positional:
            return self._positional[0].term.span
        if self._named:
            return next(iter(self._named.values())).term.span
        return self._span

    def remaining_positional(self) -> list[EArg]:
        return list(self._positional)

    def remaining_named(self) -> Iterable[EArg]:
        return self._named.values()
