"""Elaboration AST (normalized surface core)."""

from __future__ import annotations

from dataclasses import dataclass

from mltt.surface.sast import Span


@dataclass(frozen=True)
class ETerm:
    span: Span


@dataclass(frozen=True)
class EBinder:
    name: str
    ty: ETerm | None
    span: Span
    implicit: bool = False


@dataclass(frozen=True)
class EArg:
    term: ETerm
    implicit: bool = False
    name: str | None = None


@dataclass(frozen=True)
class EVar(ETerm):
    name: str


@dataclass(frozen=True)
class EConst(ETerm):
    name: str


@dataclass(frozen=True)
class EUniv(ETerm):
    level: int | str | None


@dataclass(frozen=True)
class EAnn(ETerm):
    term: ETerm
    ty: ETerm


@dataclass(frozen=True)
class EHole(ETerm):
    pass


@dataclass(frozen=True)
class ELam(ETerm):
    binders: tuple[EBinder, ...]
    body: ETerm


@dataclass(frozen=True)
class EPi(ETerm):
    binders: tuple[EBinder, ...]
    body: ETerm


@dataclass(frozen=True)
class EApp(ETerm):
    fn: ETerm
    args: tuple[EArg, ...]


@dataclass(frozen=True)
class EUApp(ETerm):
    head: ETerm
    levels: tuple[int | str, ...]


@dataclass(frozen=True)
class EPartial(ETerm):
    term: ETerm


@dataclass(frozen=True)
class ELet(ETerm):
    uparams: tuple[str, ...]
    name: str
    ty: ETerm | None
    val: ETerm
    body: ETerm


@dataclass(frozen=True)
class EPat:
    span: Span


@dataclass(frozen=True)
class EPatVar(EPat):
    name: str


@dataclass(frozen=True)
class EPatWild(EPat):
    pass


@dataclass(frozen=True)
class EPatCtor(EPat):
    ctor: str
    args: tuple[EPat, ...]


@dataclass(frozen=True)
class EBranch:
    pat: EPat
    rhs: ETerm
    span: Span


@dataclass(frozen=True)
class EMatch(ETerm):
    scrutinee: ETerm
    motive: ETerm | None
    branches: tuple[EBranch, ...]


@dataclass(frozen=True)
class EInd(ETerm):
    name: str


@dataclass(frozen=True)
class ECtor(ETerm):
    name: str


@dataclass(frozen=True)
class EConstructorDecl:
    name: str
    fields: tuple[EBinder, ...]
    result: ETerm
    span: Span


@dataclass(frozen=True)
class EInductiveDef(ETerm):
    name: str
    uparams: tuple[str, ...]
    params: tuple[EBinder, ...]
    level: ETerm
    ctors: tuple[EConstructorDecl, ...]
    body: ETerm
