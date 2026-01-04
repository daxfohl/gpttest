"""Elaboration error type."""

from __future__ import annotations

from dataclasses import dataclass

from mltt.surface.sast import Span


@dataclass
class ElabError(Exception):
    message: str
    span: Span
    source: str | None = None

    def __str__(self) -> str:
        if self.source is None:
            return f"{self.message} @ {self.span.start}:{self.span.end}"
        snippet = self.span.extract(self.source)
        return f"{self.message} @ {self.span.start}:{self.span.end}: {snippet!r}"
