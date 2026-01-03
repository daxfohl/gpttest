"""Surface AST dataclasses (purely structural)."""

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
    implicit: bool = False


@dataclass(frozen=True)
class SArg:
    term: SurfaceTerm
    implicit: bool = False
    name: str | None = None


@dataclass(frozen=True)
class SVar(SurfaceTerm):
    name: str


@dataclass(frozen=True)
class SConst(SurfaceTerm):
    name: str


@dataclass(frozen=True)
class SUniv(SurfaceTerm):
    level: int | str | None


@dataclass(frozen=True)
class SAnn(SurfaceTerm):
    term: SurfaceTerm
    ty: SurfaceTerm


@dataclass(frozen=True)
class SHole(SurfaceTerm):
    pass


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
    args: tuple[SArg, ...]


@dataclass(frozen=True)
class SUApp(SurfaceTerm):
    head: SurfaceTerm
    levels: tuple[int | str, ...]


@dataclass(frozen=True)
class SPartial(SurfaceTerm):
    term: SurfaceTerm


@dataclass(frozen=True)
class SLet(SurfaceTerm):
    uparams: tuple[str, ...]
    name: str
    ty: SurfaceTerm | None
    val: SurfaceTerm
    body: SurfaceTerm


@dataclass(frozen=True)
class Pat:
    span: Span


@dataclass(frozen=True)
class PatVar(Pat):
    name: str


@dataclass(frozen=True)
class PatWild(Pat):
    pass


@dataclass(frozen=True)
class PatCtor(Pat):
    ctor: str
    args: tuple[Pat, ...]


@dataclass(frozen=True)
class PatTuple(Pat):
    elts: tuple[Pat, ...]


@dataclass(frozen=True)
class SBranch:
    pat: Pat
    rhs: SurfaceTerm
    span: Span


@dataclass(frozen=True)
class SMatch(SurfaceTerm):
    scrutinees: tuple[SurfaceTerm, ...]
    as_names: tuple[str | None, ...]
    motive: SurfaceTerm | None
    branches: tuple[SBranch, ...]


@dataclass(frozen=True)
class SLetPat(SurfaceTerm):
    pat: Pat
    value: SurfaceTerm
    body: SurfaceTerm


@dataclass(frozen=True)
class SInd(SurfaceTerm):
    name: str


@dataclass(frozen=True)
class SCtor(SurfaceTerm):
    name: str


@dataclass(frozen=True)
class SConstructorDecl:
    name: str
    fields: tuple[SBinder, ...]
    result: SurfaceTerm | None
    span: Span


@dataclass(frozen=True)
class SInductiveDef(SurfaceTerm):
    name: str
    uparams: tuple[str, ...]
    params: tuple[SBinder, ...]
    level: SurfaceTerm
    ctors: tuple[SConstructorDecl, ...]
    body: SurfaceTerm
