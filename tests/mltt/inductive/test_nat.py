import pytest

from mltt.kernel.ast import Pi
from mltt.kernel.debruijn import mk_pis
from mltt.inductive.maybe import MaybeType, Nothing, Just
from mltt.inductive.nat import (
    NatType,
    Succ,
    Zero,
    add_term,
    add,
    numeral,
    ZeroCtor,
    SuccCtor,
    pred_maybe,
)


def test_add_has_expected_pi_type() -> None:
    add_type = mk_pis(NatType(), NatType(), return_ty=NatType())

    add_term().type_check(add_type)


def test_add_zero_left_identity() -> None:
    n_term = numeral(4)
    expected = numeral(4)

    result = add(Zero(), n_term).normalize()

    assert result == expected


def test_add_satisfies_recursive_step() -> None:
    k_term = numeral(2)
    n_term = numeral(3)

    lhs = add(Succ(k_term), n_term).normalize()
    rhs = Succ(add(k_term, n_term)).normalize()

    assert lhs == rhs


def test_add_produces_expected_numeral() -> None:
    result = add(numeral(2), numeral(3)).normalize()

    assert result == numeral(5)


@pytest.mark.parametrize("i", range(3))
def test_infer_type(i: int) -> None:
    t = numeral(i).infer_type()
    assert t == NatType()


def test_ctor_type() -> None:
    t = ZeroCtor.infer_type()
    assert t == NatType()
    t = SuccCtor.infer_type()
    assert t == Pi(NatType(), NatType())


def test_pred_maybe_zero() -> None:
    result = pred_maybe(Zero())
    assert result.normalize() == Nothing(NatType())

    result.type_check(MaybeType(NatType()))


@pytest.mark.parametrize("i", range(1, 4))
def test_pred_maybe_succ(i: int) -> None:
    n = numeral(i)
    result = pred_maybe(n)
    assert result.normalize() == Just(NatType(), numeral(i - 1))

    result.type_check(MaybeType(NatType()))
