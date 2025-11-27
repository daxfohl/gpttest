"""Basic equality combinators defined via the inductive identity type."""

from __future__ import annotations

from .ast import (
    App,
    InductiveConstructor,
    InductiveElim,
    InductiveType,
    Lam,
    Term,
    Univ,
    Var,
)
from .debruijn import shift

Eq = InductiveType(param_types=(Univ(0), Var(0)), index_types=(Var(1),), level=0)
ReflCtor = InductiveConstructor(Eq, (), (Var(1),))
object.__setattr__(Eq, "constructors", (ReflCtor,))


def Id(A: Term, x: Term, y: Term) -> Term:
    return App(App(App(Eq, A), x), y)


def Refl(A: Term, x: Term) -> Term:
    return App(App(App(ReflCtor, A), x), x)


def IdElim(A: Term, x: Term, P: Term, d: Term, y: Term, p: Term) -> InductiveElim:
    """
    Identity elimination (J) built from the generic inductive eliminator.

    Args mirror the previous native IdElim constructor:
      A: Ambient type.
      x: Base point.
      P: Motive ``λy. Id A x y -> Type``.
      d: Proof of ``P x (Refl x)``.
      y: Target point.
      p: Proof of ``Id A x y`` being eliminated.
    """

    motive = Lam(Id(A, x, y), App(App(P, y), Var(0)))
    return InductiveElim(
        inductive=Eq,
        motive=motive,
        cases=[d],
        scrutinee=p,
    )


def cong3(f: Term, A: Term, B: Term, x: Term, y: Term, p: Term) -> Term:
    """Dependent congruence for arbitrary codomains."""

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
    """Standard dependent congruence."""

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
    """Non-dependent congruence (``ap``)."""

    return cong(f, A, Lam(A, B0), x, y, p)


def sym(A: Term, x: Term, y: Term, p: Term) -> Term:
    """Symmetry of identity proofs."""

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
    """Transitivity of identity proofs."""

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


__all__ = ["Eq", "Id", "Refl", "IdElim", "cong", "sym", "trans"]
