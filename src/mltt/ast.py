"""Abstract syntax tree nodes for the miniature Martin-Lof type theory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Lam:
    """Dependent lambda term with an argument type and body.

    Args:
        ty: Type of the bound argument.
        body: Term evaluated with the bound argument in scope (index 0).
    """

    ty: Term
    body: Term


@dataclass(frozen=True)
class Pi:
    """Dependent function type (Pi-type).

    Args:
        ty: Domain type.
        body: Codomain type that may refer to the bound argument (index 0).
    """

    ty: Term
    body: Term


@dataclass(frozen=True)
class App:
    """Function application.

    Args:
        func: Term expected to reduce to a function.
        arg: Argument term supplied to ``func``.
    """

    func: Term
    arg: Term


@dataclass(frozen=True)
class Univ:
    """A universe ``Type(level)``."""

    level: int = 0

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("Universe level must be non-negative")


@dataclass(frozen=True)
class InductiveConstructor:
    """A constructor for an inductive type."""

    inductive: "InductiveType"
    arg_types: Sequence["Term"]
    result_indices: Sequence["Term"] = ()


@dataclass(frozen=True)
class InductiveType:
    """A generalized inductive type with constructors."""

    param_types: Sequence["Term"] = ()
    index_types: Sequence["Term"] = ()
    constructors: Sequence[InductiveConstructor] = ()
    level: int = 0


@dataclass(frozen=True)
class InductiveElim:
    """Elimination principle for an inductive type.

    Args:
        inductive: Inductive type being eliminated.
        motive: Motive ``λx. Type``.
        cases: Branches aligned with ``inductive.constructors``.
        scrutinee: Term of the inductive type being eliminated.
    """

    inductive: InductiveType
    motive: Term
    cases: list[Term]
    scrutinee: Term


type Term = (
    Var | Lam | Pi | App | Univ | InductiveConstructor | InductiveType | InductiveElim
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
    "InductiveElim",
]
