"""Abstract syntax tree nodes for the miniature Martin-Lof type theory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias


@dataclass
class Var:
    """De Bruijn variable pointing to the binder at ``k``."""

    k: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("De Bruijn indices must be non-negative")


@dataclass
class Lam:
    """Dependent lambda term with an argument type and body."""

    ty: Term
    body: Term


@dataclass
class Pi:
    """Dependent function type (Pi-type)."""

    ty: Term
    body: Term


@dataclass
class Sigma:
    """Dependent pair type (Sigma-type)."""

    ty: Term
    body: Term


@dataclass
class Pair:
    """Dependent pair constructor."""

    fst: Term
    snd: Term


@dataclass
class App:
    """Function application."""

    func: Term
    arg: Term


@dataclass
class Univ:
    """A universe ``Type(level)``."""

    level: int = 0

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("Universe level must be non-negative")


@dataclass
class NatType:
    """The natural numbers type."""

    pass


@dataclass
class Zero:
    """Zero constructor for the natural numbers."""

    pass


@dataclass
class Succ:
    """Successor constructor for the natural numbers."""

    n: Term


@dataclass
class NatRec:
    """Primitive recursion principle for natural numbers."""

    P: Term
    base: Term
    step: Term
    n: Term


@dataclass
class Id:
    """Identity type over ``ty`` relating ``lhs`` and ``rhs``."""

    ty: Term
    lhs: Term
    rhs: Term


@dataclass
class Refl:
    """Canonical inhabitant of an identity type."""

    ty: Term
    t: Term


@dataclass
class IdElim:
    """Identity elimination principle (J)."""

    A: Term
    x: Term
    P: Term
    d: Term
    y: Term
    p: Term


Term: TypeAlias = (
    Var
    | Lam
    | Pi
    | Sigma
    | Pair
    | App
    | Univ
    | NatType
    | Zero
    | Succ
    | NatRec
    | Id
    | Refl
    | IdElim
)


__all__ = [
    "Term",
    "Var",
    "Lam",
    "Pi",
    "Sigma",
    "Pair",
    "App",
    "Univ",
    "NatType",
    "Zero",
    "Succ",
    "NatRec",
    "Id",
    "Refl",
    "IdElim",
]
