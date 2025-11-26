"""Abstract syntax tree nodes for the miniature Martin-Lof type theory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass
class Var:
    """De Bruijn variable pointing to the binder at ``k``.

    Args:
        k: Zero-based index counting binders outward from the binding site.
           ``0`` refers to the innermost binder, ``1`` to the next, etc.
    """

    k: int

    def __post_init__(self) -> None:
        if self.k < 0:
            raise ValueError("De Bruijn indices must be non-negative")


@dataclass
class Lam:
    """Dependent lambda term with an argument type and body.

    Args:
        ty: Type of the bound argument.
        body: Term evaluated with the bound argument in scope (index 0).
    """

    ty: Term
    body: Term


@dataclass
class Pi:
    """Dependent function type (Pi-type).

    Args:
        ty: Domain type.
        body: Codomain type that may refer to the bound argument (index 0).
    """

    ty: Term
    body: Term


@dataclass
class App:
    """Function application.

    Args:
        func: Term expected to reduce to a function.
        arg: Argument term supplied to ``func``.
    """

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
class InductiveConstructor:
    """A constructor for an inductive type."""

    name: str
    arg_types: Sequence["Term"]


@dataclass
class InductiveType:
    """A generalized inductive type with named constructors."""

    name: str
    constructors: Sequence[InductiveConstructor] = ()
    level: int = 0


@dataclass
class ConstructorApp:
    """Application of a constructor to its arguments."""

    inductive: InductiveType
    constructor: str
    args: Sequence["Term"]


@dataclass
class InductiveElim:
    """Elimination principle for an inductive type.

    Args:
        inductive: Inductive type being eliminated.
        motive: Motive ``λx. Type``.
        cases: Mapping from constructor name to case branch.
        scrutinee: Term of the inductive type being eliminated.
    """

    inductive: InductiveType
    motive: Term
    cases: Mapping[str, Term]
    scrutinee: Term


@dataclass
class Id:
    """Identity type over ``ty`` relating ``lhs`` and ``rhs``.

    Args:
        ty: Ambient type ``A``.
        lhs: Left endpoint ``x``.
        rhs: Right endpoint ``y``.
    """

    ty: Term
    lhs: Term
    rhs: Term


@dataclass
class Refl:
    """Canonical inhabitant of an identity type.

    Args:
        ty: Ambient type ``A``.
        t: Witness term ``x``; produces ``Id A x x``.
    """

    ty: Term
    t: Term


@dataclass
class IdElim:
    """Identity elimination principle (J).

    Args:
        A: Ambient type ``A``.
        x: Base point ``x : A``.
        P: Motive ``λy. Id A x y -> Type``.
        d: Proof of ``P x (Refl x)``.
        y: Target point ``y : A``.
        p: Proof of ``Id A x y`` being eliminated.
    """

    A: Term
    x: Term
    P: Term
    d: Term
    y: Term
    p: Term


type Term = (
    Var
    | Lam
    | Pi
    | App
    | Univ
    | InductiveType
    | ConstructorApp
    | InductiveElim
    | Id
    | Refl
    | IdElim
)


__all__ = [
    "Term",
    "Var",
    "Lam",
    "Pi",
    "App",
    "Univ",
    "InductiveConstructor",
    "InductiveType",
    "ConstructorApp",
    "InductiveElim",
    "Id",
    "Refl",
    "IdElim",
]
