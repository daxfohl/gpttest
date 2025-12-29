from mltt.kernel.ast import App, Lam, Var
from mltt.kernel.telescope import mk_lams
from mltt.inductive.eq import Id, IdElim, Refl, cong, sym, trans
from mltt.inductive.nat import NatType, Succ, Zero


def test_cong_builds_identity_elimination_over_function_application() -> None:
    A = NatType()
    B = Lam(NatType(), Var(0))
    f = Lam(NatType(), Succ(Var(0)))
    x = Zero()
    p = Var(1)

    result = cong(f, A, B, x, p)

    P = mk_lams(
        A,
        Id(A, x, Var(0)),
        body=Id(App(B, Var(1)), App(f, x), App(f, Var(1))),
    )
    d = Refl(App(B, x), App(f, x))
    expected = IdElim(P=P, d=d, p=p)

    assert result == expected


def test_sym_builds_identity_elimination_with_swapped_arguments() -> None:
    A = NatType()
    x = Zero()
    p = Var(0)

    result = sym(A, x, p)

    P = mk_lams(
        A,
        Id(A, x, Var(0)),
        body=Id(A, Var(1), x),
    )
    d = Refl(A, x)
    expected = IdElim(P=P, d=d, p=p)

    assert result == expected


def test_trans_builds_identity_elimination_for_composition() -> None:
    A = NatType()
    x = Zero()
    y = Succ(Zero())
    p = Var(0)
    q = Var(1)

    result = trans(A, x, y, p, q)

    Q = mk_lams(
        A,
        Id(A, y, Var(0)),
        body=Id(A, x, Var(1)),
    )
    expected = IdElim(P=Q, d=p, p=q)

    assert result == expected
