from mltt.ast import App, Id, NatType, Pi, Refl, Succ, Var, Zero
from mltt.eval import normalize
from mltt.nat import add, numeral
from mltt.typing import type_check


def test_add_has_expected_pi_type():
    add_type = Pi(NatType(), Pi(NatType(), NatType()))

    assert type_check(add(), add_type)


def test_add_zero_left_identity():
    n_term = numeral(4)
    expected = numeral(4)

    result = normalize(App(App(add(), Zero()), n_term))

    assert result == expected


def test_add_satisfies_recursive_step():
    k_term = numeral(2)
    n_term = numeral(3)

    lhs = normalize(App(App(add(), Succ(k_term)), n_term))
    rhs = normalize(Succ(App(App(add(), k_term), n_term)))

    assert lhs == rhs


def test_add_produces_expected_numeral():
    result = normalize(App(App(add(), numeral(2)), numeral(3)))

    assert result == numeral(5)
