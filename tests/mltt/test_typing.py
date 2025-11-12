import pytest

from mltt.ast import (
    App,
    Id,
    IdElim,
    Lam,
    NatRec,
    NatType,
    Pair,
    Pi,
    Refl,
    Sigma,
    Univ,
    Var,
    Zero,
)
from mltt.typing import infer_type, type_check, type_equal
from mltt.nat import add, numeral


def test_type_equal_normalizes_beta_equivalent_terms() -> None:
    beta_equiv = App(Lam(Univ(), Var(0)), Univ())

    assert type_equal(beta_equiv, Univ())
    assert not type_equal(beta_equiv, NatType())


def test_type_universe_levels_are_indexed() -> None:
    assert infer_type(Univ()) == Univ(1)
    assert infer_type(Univ(2)) == Univ(3)


def test_infer_type_of_lambda_returns_pi_type() -> None:
    term = Lam(NatType(), Var(0))

    assert infer_type(term) == Pi(NatType(), NatType())


def test_infer_type_of_pi_uses_maximum_universe_level() -> None:
    assert infer_type(Pi(NatType(), NatType())) == Univ(0)
    higher = Pi(Univ(), NatType())
    assert infer_type(higher) == Univ(1)
    cod_dominates = Pi(NatType(), Univ(1))
    assert infer_type(cod_dominates) == Univ(2)


def test_infer_type_application_requires_function() -> None:
    with pytest.raises(TypeError, match="Application of non-function"):
        infer_type(App(Zero(), Zero()))


def test_type_check_pair_against_sigma_type() -> None:
    pair = Pair(Zero(), NatType())
    sigma_ty = Sigma(NatType(), Univ())

    assert type_check(pair, sigma_ty)


def test_type_check_natrec_rejects_invalid_base_case() -> None:
    P = Lam(NatType(), NatType())
    z = Univ()
    s = Zero()
    n = Zero()
    term = NatRec(P, z, s, n)

    with pytest.raises(TypeError, match="NatRec base case type mismatch"):
        type_check(term, App(P, n))


def test_type_check_accepts_add_application() -> None:
    term = App(App(add(), numeral(2)), numeral(3))

    assert type_check(term, NatType())


def test_type_check_lambda_with_wrong_domain() -> None:
    term = Lam(NatType(), Var(0))
    expected = Pi(Univ(), NatType())
    with pytest.raises(TypeError, match="Lambda domain mismatch"):
        type_check(term, expected)


def test_type_check_application_argument_mismatch() -> None:
    f = Lam(NatType(), Var(0))
    term = App(f, Univ())
    with pytest.raises(TypeError, match="Application argument type mismatch"):
        type_check(term, NatType())


def test_infer_type_idelim() -> None:
    term = IdElim(
        Univ(),
        Var(0),
        Lam(Univ(), Lam(Id(Univ(), Var(0), Var(1)), Univ())),
        Var(0),
        Var(1),
        Refl(Univ(), Var(0)),
    )
    inferred = infer_type(term)
    assert inferred == App(
        App(
            Lam(Univ(), Lam(Id(Univ(), Var(0), Var(1)), Univ())),
            Var(1),
        ),
        Refl(Univ(), Var(0)),
    )
