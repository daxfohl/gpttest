import mltt.inductive.sigma as sigma
from mltt.core.ast import App, Lam, Pi, Univ, Var
from mltt.core.inductive_utils import nested_lam, nested_pi
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_check
from mltt.inductive.nat import NatType, Succ, Zero


def test_infer_sigma_type_constructor() -> None:
    expected = nested_pi(Univ(0), Pi(Var(0), Univ(0)), return_ty=Univ(0))

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

    P = nested_lam(
        Univ(0),
        Pi(Var(0), Univ(0)),
        sigma.SigmaType(Var(1), Var(0)),
        body=Var(2),
    )
    pair_case = nested_lam(
        A,
        App(B, Var(0)),  # b : B a
        body=Var(1),  # return the first projection
    )

    fst = sigma.SigmaRec(P, pair_case, pair)

    assert type_check(fst, A)
    assert normalize(fst) == Succ(Zero())
