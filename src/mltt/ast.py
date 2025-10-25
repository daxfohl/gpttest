from __future__ import annotations

from dataclasses import dataclass


class Term:
    pass


@dataclass
class Var(Term):
    index: int


@dataclass
class Lam(Term):
    ty: Term
    body: Term


@dataclass
class Pi(Term):
    ty: Term
    body: Term


@dataclass
class Sigma(Term):
    ty: Term
    body: Term


@dataclass
class Pair(Term):
    fst: Term
    snd: Term


@dataclass
class App(Term):
    func: Term
    arg: Term


@dataclass
class TypeUniverse(Term):
    pass


@dataclass
class NatType(Term):
    pass


@dataclass
class Zero(Term):
    pass


@dataclass
class Succ(Term):
    n: Term


@dataclass
class NatRec(Term):
    P: Term
    z: Term
    s: Term
    n: Term


@dataclass
class Id(Term):
    ty: Term
    lhs: Term
    rhs: Term


@dataclass
class Refl(Term):
    ty: Term
    t: Term


@dataclass
class IdElim(Term):
    A: Term
    x: Term
    P: Term
    d: Term
    y: Term
    p: Term


__all__ = [
    "Term",
    "Var",
    "Lam",
    "Pi",
    "Sigma",
    "Pair",
    "App",
    "TypeUniverse",
    "NatType",
    "Zero",
    "Succ",
    "NatRec",
    "Id",
    "Refl",
    "IdElim",
]
