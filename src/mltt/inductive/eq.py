"""Basic equality combinators for the identity type."""

from __future__ import annotations

from mltt.core.ast import App, Lam, Term, Univ, Var
from mltt.core.debruijn import mk_app, mk_lams, Telescope, ArgList
from mltt.core.ind import Elim, Ctor, Ind

IdType = Ind(
    name="Id",
    param_types=Telescope.of(Univ(0), Var(0)),
    index_types=Telescope.of(Var(1)),
    level=0,
)
ReflCtor = Ctor(
    name="Refl",
    inductive=IdType,
    result_indices=ArgList.of(Var(0)),
)
object.__setattr__(IdType, "constructors", (ReflCtor,))


def Id(ty: Term, lhs: Term, rhs: Term) -> Term:
    """Identity type over ``ty`` relating ``lhs`` and ``rhs``."""

    return mk_app(IdType, ty, lhs, rhs)


def Refl(ty: Term, t: Term) -> Term:
    """Canonical inhabitant ``Id ty t t``."""

    return mk_app(ReflCtor, ty, t)


def IdElim(P: Term, d: Term, p: Term) -> Elim:
    """Identity eliminator (J) expressed via the generalized ``Elim``."""

    return Elim(inductive=IdType, motive=P, cases=(d,), scrutinee=p)


def cong3(f: Term, A: Term, B: Term, x: Term, p: Term) -> Term:
    """Dependent congruence for arbitrary codomains."""

    P = mk_lams(
        A,
        Id(A.shift(1), x.shift(1), Var(0)),
        body=Id(
            App(B.shift(2), Var(1)),
            App(f.shift(2), x.shift(2)),
            App(f.shift(2), Var(1)),
        ),
    )
    d = Refl(App(B, x), App(f, x))
    return IdElim(P, d, p)


def cong(f: Term, A: Term, B: Term, x: Term, p: Term) -> Term:
    """Standard dependent congruence."""

    A1 = A.shift(1)
    x1 = x.shift(1)
    B2 = B.shift(2)
    f2 = f.shift(2)
    x2 = x.shift(2)

    P = mk_lams(
        A,
        Id(A1, x1, Var(0)),
        body=Id(App(B2, Var(1)), App(f2, x2), App(f2, Var(1))),
    )
    d = Refl(App(B, x), App(f, x))
    return IdElim(P, d, p)


def ap(f: Term, A: Term, B0: Term, x: Term, p: Term) -> Term:
    """Non-dependent congruence (``ap``)."""

    return cong(f, A, Lam(A, B0), x, p)


def sym(A: Term, x: Term, p: Term) -> Term:
    """Symmetry of identity proofs."""

    A1 = A.shift(1)
    x1 = x.shift(1)
    A2 = A.shift(2)
    x2 = x.shift(2)

    P = mk_lams(
        A,
        Id(A1, x1, Var(0)),
        body=Id(A2, Var(1), x2),
    )
    d = Refl(A, x)
    return IdElim(P, d, p)


def trans(A: Term, x: Term, y: Term, p: Term, q: Term) -> Term:
    """Transitivity of identity proofs."""

    A1 = A.shift(1)
    y1 = y.shift(1)
    A2 = A.shift(2)
    x2 = x.shift(2)

    Q = mk_lams(
        A,
        Id(A1, y1, Var(0)),
        body=Id(A2, x2, Var(1)),
    )
    return IdElim(Q, p, q)
