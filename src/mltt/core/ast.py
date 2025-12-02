"""Abstract syntax tree nodes for the miniature Martin-Lof type theory."""

from __future__ import annotations

from dataclasses import dataclass


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
class Ctor:
    """A constructor for an inductive type."""

    name: str
    inductive: I
    arg_types: tuple[Term, ...]
    result_indices: tuple[Term, ...] = ()


@dataclass(frozen=True)
class I:
    """A generalized inductive type with constructors.

    Args:
        name: Human-readable identifier used by pretty-printers.
    """

    name: str
    param_types: tuple[Term, ...] = ()
    index_types: tuple[Term, ...] = ()
    constructors: tuple[Ctor, ...] = ()
    level: int = 0


@dataclass(frozen=True)
class Elim:
    """Elimination principle for an inductive type.

    Args:
        inductive: Inductive type being eliminated.
        motive: Motive ``λx. Type``.
        cases: Branches aligned with ``inductive.constructors``.
        scrutinee: Term of the inductive type being eliminated.
    """

    inductive: I
    motive: Term
    cases: list[Term]
    scrutinee: Term


@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Refl:
    """Canonical inhabitant of an identity type.

    Args:
        ty: Ambient type ``A``.
        t: Witness term ``x``; produces ``Id A x x``.
    """

    ty: Term
    t: Term


@dataclass(frozen=True)
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
        | Ctor
        | I
        | Elim
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
    "Ctor",
    "I",
    "Elim",
    "Id",
    "Refl",
    "IdElim",
]


def _repr(self: Term) -> str:
    # Deferred import to avoid cycles when pretty-printing from dataclass repr.
    from .pretty import pretty

    return pretty(self)


for _cls in (
        Var,
        Lam,
        Pi,
        App,
        Univ,
        Ctor,
        I,
        Elim,
        Id,
        Refl,
        IdElim,
):
    _cls.__repr__ = _repr  # type: ignore
