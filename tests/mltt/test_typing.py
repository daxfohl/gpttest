import pytest

from mltt.ast import (
    App,
    Lam,
    NatRec,
    NatType,
    Pair,
    Pi,
    Sigma,
    TypeUniverse,
    Var,
    Zero,
)
from mltt.typing import infer_type, type_check, type_equal
from mltt.nat import add, numeral


def test_type_equal_normalizes_beta_equivalent_terms():
    beta_equiv = App(Lam(TypeUniverse(), Var(0)), TypeUniverse())

    assert type_equal(beta_equiv, TypeUniverse())
    assert not type_equal(beta_equiv, NatType())


def test_infer_type_of_lambda_returns_pi_type():
    term = Lam(NatType(), Var(0))

    assert infer_type(term) == Pi(NatType(), NatType())


def test_infer_type_application_requires_function():
    with pytest.raises(TypeError, match="Application of non-function"):
        infer_type(App(Zero(), Zero()))


def test_type_check_pair_against_sigma_type():
    pair = Pair(Zero(), NatType())
    sigma_ty = Sigma(NatType(), TypeUniverse())

    assert type_check(pair, sigma_ty)


def test_type_check_natrec_rejects_invalid_base_case():
    P = Lam(NatType(), NatType())
    z = TypeUniverse()
    s = Zero()
    n = Zero()
    term = NatRec(P, z, s, n)

    with pytest.raises(TypeError, match="NatRec base case type mismatch"):
        type_check(term, App(P, n))


def test_type_check_accepts_add_application():
    term = App(App(add, numeral(2)), numeral(3))

    assert type_check(term, NatType())
