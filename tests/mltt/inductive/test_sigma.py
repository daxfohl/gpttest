import mltt.inductive.sigma as sigma
from mltt.core.ast import App, Lam, Pi, Univ, Var
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_check
from mltt.inductive.nat import NatType, Succ, Zero


def test_infer_sigma_type_constructor() -> None:
    expected = Pi(Univ(0), Pi(Pi(Var(0), Univ(0)), Univ(0)))

    assert infer_type(sigma.Sigma) == expected


def test_pair_type_check() -> None:
    A = NatType()
    B = Lam(A, NatType())
    pair = sigma.Pair(A, B, Zero(), Zero())

    assert type_check(pair, sigma.SigmaType(A, B))


def test_sigmarec_returns_first_projection() -> None:
    A = NatType()
    B = Lam(A, NatType())

    pair = sigma.Pair(A, B, Succ(Zero()), Zero())

    P = Lam(
        Univ(0), Lam(Pi(Var(0), Univ(0)), Lam(sigma.SigmaType(Var(1), Var(0)), Var(2)))
    )
    pair_case = Lam(
        A,
        Lam(
            App(B, Var(0)),  # b : B a
            Var(1),  # return the first projection
        ),
    )

    fst = sigma.SigmaRec(P, pair_case, pair)

    assert type_check(fst, A)
    assert normalize(fst) == Succ(Zero())
