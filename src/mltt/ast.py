"""Abstract syntax tree nodes for the miniature Martin-Lof type theory."""

from __future__ import annotations

from dataclasses import dataclass


class Term:
    """Base class for all core syntax tree nodes."""

    pass


@dataclass
class Var(Term):
    """De Bruijn variable pointing to the binder at ``index``."""

    index: int


@dataclass
class Lam(Term):
    """Dependent lambda term with an argument type and body."""

    ty: Term
    body: Term


@dataclass
class Pi(Term):
    """Dependent function type (Pi-type)."""

    ty: Term
    body: Term


@dataclass
class Sigma(Term):
    """Dependent pair type (Sigma-type)."""

    ty: Term
    body: Term


@dataclass
class Pair(Term):
    """Dependent pair constructor."""

    fst: Term
    snd: Term


@dataclass
class App(Term):
    """Function application."""

    func: Term
    arg: Term


@dataclass
class TypeUniverse(Term):
    """The sole universe used by the miniature theory."""

    pass


@dataclass
class NatType(Term):
    """The natural numbers type."""

    pass


@dataclass
class Zero(Term):
    """Zero constructor for the natural numbers."""

    pass


@dataclass
class Succ(Term):
    """Successor constructor for the natural numbers."""

    n: Term


@dataclass
class NatRec(Term):
    """Primitive recursion principle for natural numbers."""

    P: Term
    z: Term
    s: Term
    n: Term


@dataclass
class Id(Term):
    """Identity type over ``ty`` relating ``lhs`` and ``rhs``."""

    ty: Term
    lhs: Term
    rhs: Term


@dataclass
class Refl(Term):
    """Canonical inhabitant of an identity type."""

    ty: Term
    t: Term


@dataclass
class IdElim(Term):
    """Identity elimination principle (J)."""

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
