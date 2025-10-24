from __future__ import annotations

from .ast import App, Id, IdElim, Lam, NatType, Refl, Succ, Term, Var


def cong(f: Term, A: Term, B: Term, x: Term, y: Term, p: Term) -> Term:
    P = Lam(Lam(Id(App(B, Var(1)), App(f, x), App(f, Var(1)))))
    d = Refl(App(B, x), App(f, x))
    return IdElim(A, x, P, d, y, p)


def sym(A: Term, x: Term, y: Term, p: Term) -> Term:
    P = Lam(Lam(Id(A, Var(1), x)))
    d = Refl(A, x)
    return IdElim(A, x, P, d, y, p)


def trans(A: Term, x: Term, y: Term, z: Term, p: Term, q: Term) -> Term:
    Q = Lam(Lam(Id(A, x, Var(1))))
    return IdElim(A, y, Q, p, z, q)


__all__ = ["cong", "sym", "trans"]
