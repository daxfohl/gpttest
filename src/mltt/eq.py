"""Basic equality combinators for the identity type."""

from __future__ import annotations

from .ast import App, Id, IdElim, Lam, Refl, Term, Var
from .debruijn import shift


def cong3(f: Term, A: Term, B: Term, x: Term, y: Term, p: Term) -> Term:
    """Dependent congruence for arbitrary codomains.

    Args:
        f: Dependent function ``(a : A) -> B a`` whose action on equal terms we lift.
        A: Domain type of ``f`` and the type witnessing ``p``.
        B: Dependent codomain family over ``A``.
        x: Left endpoint of the given equality proof.
        y: Right endpoint of the given equality proof.
        p: Proof of ``Id A x y``.

    Returns:
        A term of type ``Id (B y) (f x) (f y)`` justifying that ``f`` preserves ``p``.
    """

    P = Lam(
        A,
        shift(
            Lam(
                Id(A, x, Var(1)),
                shift(Id(App(B, Var(1)), App(f, x), App(f, Var(1))), 1),
            ),
            1,
        ),
    )
    d = Refl(App(B, x), App(f, x))
    return IdElim(A, x, P, d, y, p)


def cong(f: Term, A: Term, B: Term, x: Term, y: Term, p: Term) -> Term:
    """Standard dependent congruence.

    Args:
        f: Dependent function ``(a : A) -> B a``.
        A: Domain type.
        B: Codomain family depending on ``A``.
        x: Left endpoint of ``p``.
        y: Right endpoint of ``p``.
        p: Proof of ``Id A x y``.

    Returns:
        Proof of ``Id (B y) (f x) (f y)`` obtained by lifting ``p`` through ``f``.
    """

    A1 = shift(A, 1)
    x1 = shift(x, 1)
    A2 = shift(A, 2)
    B2 = shift(B, 2)
    f2 = shift(f, 2)
    x2 = shift(x, 2)

    P = Lam(
        A,
        Lam(
            Id(A1, x1, Var(1)),
            Id(App(B2, Var(1)), App(f2, x2), App(f2, Var(1))),
        ),
    )
    d = Refl(App(B, x), App(f, x))
    return IdElim(A, x, P, d, y, p)


def ap(f: Term, A: Term, B0: Term, x: Term, y: Term, p: Term) -> Term:
    """Non-dependent congruence (``ap``).

    Args:
        f: Plain function ``A -> B0``.
        A: Domain type.
        B0: Codomain type (constant family).
        x: Left endpoint of ``p``.
        y: Right endpoint of ``p``.
        p: Proof of ``Id A x y``.

    Returns:
        Proof of ``Id B0 (f x) (f y)`` asserting ``f`` preserves equality.
    """

    return cong(f, A, Lam(A, B0), x, y, p)


def sym(A: Term, x: Term, y: Term, p: Term) -> Term:
    """Symmetry of identity proofs.

    Args:
        A: Ambient type.
        x: Left endpoint.
        y: Right endpoint.
        p: Proof of ``Id A x y``.

    Returns:
        A proof of ``Id A y x`` obtained by flipping ``p``.
    """

    A1 = shift(A, 1)
    x1 = shift(x, 1)
    A2 = shift(A, 2)
    x2 = shift(x, 2)

    P = Lam(
        A,
        Lam(
            Id(A1, x1, Var(1)),
            Id(A2, Var(1), x2),
        ),
    )
    d = Refl(A, x)
    return IdElim(A, x, P, d, y, p)


def trans(A: Term, x: Term, y: Term, z: Term, p: Term, q: Term) -> Term:
    """Transitivity of identity proofs.

    Args:
        A: Ambient type.
        x: First element.
        y: Middle element shared between the two proofs.
        z: Final element.
        p: Proof of ``Id A x y``.
        q: Proof of ``Id A y z``.

    Returns:
        A proof of ``Id A x z`` composing ``p`` and ``q``.
    """

    A1 = shift(A, 1)
    y1 = shift(y, 1)
    A2 = shift(A, 2)
    x2 = shift(x, 2)

    Q = Lam(
        A,
        Lam(
            Id(A1, y1, Var(1)),
            Id(A2, x2, Var(1)),
        ),
    )
    return IdElim(A, y, Q, p, z, q)


__all__ = ["cong", "sym", "trans"]
