import pytest

from mltt.core.ast import Pi
from mltt.core.inductive_utils import nested_pi
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_check
from mltt.inductive.maybe import MaybeType, Nothing, Just
from mltt.inductive.nat import (
    NatType,
    Succ,
    Zero,
    add,
    add_terms,
    numeral,
    ZeroCtor,
    SuccCtor,
    pred_maybe_terms,
)


def test_add_has_expected_pi_type() -> None:
    add_type = nested_pi(NatType(), NatType(), return_ty=NatType())

    assert type_check(add(), add_type)


def test_add_zero_left_identity() -> None:
    n_term = numeral(4)
    expected = numeral(4)

    result = normalize(add_terms(Zero(), n_term))

    assert result == expected


def test_add_satisfies_recursive_step() -> None:
    k_term = numeral(2)
    n_term = numeral(3)

    lhs = normalize(add_terms(Succ(k_term), n_term))
    rhs = normalize(Succ(add_terms(k_term, n_term)))

    assert lhs == rhs


def test_add_produces_expected_numeral() -> None:
    result = normalize(add_terms(numeral(2), numeral(3)))

    assert result == numeral(5)


@pytest.mark.parametrize("i", range(3))
def test_infer_type(i: int) -> None:
    t = infer_type(numeral(i))
    assert t == NatType()


def test_ctor_type() -> None:
    t = infer_type(ZeroCtor)
    assert t == NatType()
    t = infer_type(SuccCtor)
    assert t == Pi(NatType(), NatType())


def test_pred_maybe_zero() -> None:
    result = pred_maybe_terms(Zero())
    assert normalize(result) == Nothing(NatType())
    assert type_check(result, MaybeType(NatType()))


@pytest.mark.parametrize("i", range(1, 4))
def test_pred_maybe_succ(i: int) -> None:
    n = numeral(i)
    result = pred_maybe_terms(n)
    assert normalize(result) == Just(NatType(), numeral(i - 1))
    assert type_check(result, MaybeType(NatType()))
