"""Basic equality combinators for the identity type."""

from __future__ import annotations

from ..core.ast import App, Ctor, Elim, I, Lam, Term, Univ, Var
from ..core.debruijn import shift
from ..core.util import apply_term, nested_lam

IdType = I(
    name="Id",
    param_types=(Univ(0), Var(0)),
    index_types=(Var(1),),
    level=0,
)
ReflCtor = Ctor(
    name="Refl",
    inductive=IdType,
    arg_types=(),
    result_indices=(Var(0),),
)
object.__setattr__(IdType, "constructors", (ReflCtor,))


def Id(ty: Term, lhs: Term, rhs: Term) -> Term:
    """Identity type over ``ty`` relating ``lhs`` and ``rhs``."""

    return apply_term(IdType, ty, lhs, rhs)


def Refl(ty: Term, t: Term) -> Term:
    """Canonical inhabitant ``Id ty t t``."""

    return apply_term(ReflCtor, ty, t)


def IdElim(A: Term, x: Term, P: Term, d: Term, y: Term, p: Term) -> Elim:
    """Identity eliminator (J) expressed via the generalized ``Elim``."""

    # Parameters ``A`` and ``x`` match the prior eliminator signature even
    # though they are implicitly captured by ``P``/``d``.
    return Elim(inductive=IdType, motive=P, cases=(d,), scrutinee=p)


def cong3(f: Term, A: Term, B: Term, x: Term, y: Term, p: Term) -> Term:
    """Dependent congruence for arbitrary codomains."""

    P = nested_lam(
        A,
        Id(shift(A, 1), shift(x, 1), Var(0)),
        body=Id(
            App(shift(B, 2), Var(1)),
            App(shift(f, 2), shift(x, 2)),
            App(shift(f, 2), Var(1)),
        ),
    )
    d = Refl(App(B, x), App(f, x))
    return IdElim(A, x, P, d, y, p)


def cong(f: Term, A: Term, B: Term, x: Term, y: Term, p: Term) -> Term:
    """Standard dependent congruence."""

    A1 = shift(A, 1)
    x1 = shift(x, 1)
    B2 = shift(B, 2)
    f2 = shift(f, 2)
    x2 = shift(x, 2)

    P = nested_lam(
        A,
        Id(A1, x1, Var(0)),
        body=Id(App(B2, Var(1)), App(f2, x2), App(f2, Var(1))),
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

    P = nested_lam(
        A,
        Id(A1, x1, Var(0)),
        body=Id(A2, Var(1), x2),
    )
    d = Refl(A, x)
    return IdElim(A, x, P, d, y, p)


def trans(A: Term, x: Term, y: Term, z: Term, p: Term, q: Term) -> Term:
    """Transitivity of identity proofs."""

    A1 = shift(A, 1)
    y1 = shift(y, 1)
    A2 = shift(A, 2)
    x2 = shift(x, 2)

    Q = nested_lam(
        A,
        Id(A1, y1, Var(0)),
        body=Id(A2, x2, Var(1)),
    )
    return IdElim(A, y, Q, p, z, q)


__all__ = ["IdType", "Id", "Refl", "IdElim", "cong3", "cong", "ap", "sym", "trans"]
