"""Basic equality combinators for the identity type."""

from __future__ import annotations

from .ast import App, Id, IdElim, Lam, Refl, Term, Var
from .debruijn import shift


def cong3(f: Term, A: Term, B: Term, x: Term, y: Term, p: Term) -> Term:
    """Map equality along ``f`` to obtain ``f x == f y``."""

    P = Lam(
        A,
        shift(Lam(
            Id(A, x, Var(1)),
            shift(Id(App(B, Var(1)), App(f, x), App(f, Var(1))), 1),
        ), 1),
    )
    d = Refl(App(B, x), App(f, x))
    return IdElim(A, x, P, d, y, p)

def cong(f: Term, A: Term, B: Term, x: Term, y: Term, p: Term) -> Term:
    """Map equality along ``f`` to obtain ``f x == f y``."""

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
    """Non-dependent congruence (ap) as a thin wrapper (B made constant)."""

    return cong(f, A, Lam(A, B0), x, y, p)


def sym(A: Term, x: Term, y: Term, p: Term) -> Term:
    """Flip an equality proof so that ``x == y`` becomes ``y == x``."""

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
    """Compose equality proofs for transitivity."""

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
