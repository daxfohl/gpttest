"""Basic equality combinators for the identity type."""

from __future__ import annotations

from .ast import App, Id, IdElim, Lam, Refl, Term, Var


def cong(f: Term, A: Term, B: Term, x: Term, y: Term, p: Term) -> Term:
    """Map equality along ``f`` to obtain ``f x == f y``."""

    P = Lam(
        A,
        Lam(
            Id(A, x, Var(1)),
            Id(App(B, Var(1)), App(f, x), App(f, Var(1))),
        ),
    )
    d = Refl(App(B, x), App(f, x))
    return IdElim(A, x, P, d, y, p)


def sym(A: Term, x: Term, y: Term, p: Term) -> Term:
    """Flip an equality proof so that ``x == y`` becomes ``y == x``."""

    P = Lam(
        A,
        Lam(
            Id(A, x, Var(1)),
            Id(A, Var(1), x),
        ),
    )
    d = Refl(A, x)
    return IdElim(A, x, P, d, y, p)


def trans(A: Term, x: Term, y: Term, z: Term, p: Term, q: Term) -> Term:
    """Compose equality proofs for transitivity."""

    Q = Lam(
        A,
        Lam(
            Id(A, y, Var(1)),
            Id(A, x, Var(1)),
        ),
    )
    return IdElim(A, y, Q, p, z, q)


__all__ = ["cong", "sym", "trans"]
