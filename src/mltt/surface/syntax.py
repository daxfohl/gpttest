"""Surface syntax AST and error helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Span:
    start: int
    end: int

    def extract(self, source: str) -> str:
        return source[self.start : self.end]


@dataclass
class SurfaceError(Exception):
    message: str
    span: Span
    source: str | None = None

    def __str__(self) -> str:
        if self.source is None:
            return f"{self.message} @ {self.span.start}:{self.span.end}"
        snippet = self.span.extract(self.source)
        return f"{self.message} @ {self.span.start}:{self.span.end}: {snippet!r}"


@dataclass(frozen=True)
class SurfaceTerm:
    span: Span


@dataclass(frozen=True)
class SBinder:
    name: str
    ty: SurfaceTerm | None
    span: Span


@dataclass(frozen=True)
class SVar(SurfaceTerm):
    name: str


@dataclass(frozen=True)
class SType(SurfaceTerm):
    level: int


@dataclass(frozen=True)
class SAnn(SurfaceTerm):
    term: SurfaceTerm
    ty: SurfaceTerm


@dataclass(frozen=True)
class SLam(SurfaceTerm):
    binders: tuple[SBinder, ...]
    body: SurfaceTerm


@dataclass(frozen=True)
class SPi(SurfaceTerm):
    binders: tuple[SBinder, ...]
    body: SurfaceTerm


@dataclass(frozen=True)
class SApp(SurfaceTerm):
    fn: SurfaceTerm
    args: tuple[SurfaceTerm, ...]


@dataclass(frozen=True)
class SLet(SurfaceTerm):
    name: str
    ty: SurfaceTerm
    val: SurfaceTerm
    body: SurfaceTerm
