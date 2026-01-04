"""Argument matching helpers for applications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

from mltt.elab.east import EArg, ENamedArg
from mltt.elab.etype import ElabBinderInfo
from mltt.elab.errors import ElabError
from mltt.surface.sast import Span


ArgKind = Literal["explicit", "implicit", "missing", "stop"]


@dataclass(frozen=True)
class ArgDecision:
    kind: ArgKind
    arg: EArg | ENamedArg | None = None
    from_named: bool = False


class ArgMatcher:
    def __init__(
        self,
        binders: tuple[ElabBinderInfo, ...],
        positional: tuple[EArg, ...],
        named: tuple[ENamedArg, ...],
        span: Span,
    ) -> None:
        self._span = span
        self._positional = list(positional)
        named_map: dict[str, ENamedArg] = {}
        for item in named:
            named_map[item.name] = item
        if named_map and not any(b.name for b in binders):
            raise ElabError("Named arguments require binder names", span)
        self._named = named_map

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
                raise ElabError(
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
        raise ElabError("Missing explicit argument", self._span)

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

    def remaining_named(self) -> Iterable[ENamedArg]:
        return self._named.values()
