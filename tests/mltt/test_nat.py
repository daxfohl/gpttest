from mltt.ast import App, Id, NatType, Pi, Refl, Succ, Var, Zero, Term
from mltt.beta_reduce import normalize
from mltt.nat import add, numeral, add_n_0
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


def test_add_zero_right_typechecks():
    lemma = add_n_0()
    lemma_ty = Pi(
        NatType(),
        Id(NatType(), App(App(add(), Var(0)), Zero()), Var(0)),
    )

    assert type_check(lemma, lemma_ty)


def test_add_zero_right_normalizes():
    lemma = add_n_0()
    applied = App(lemma, numeral(5))
    assert normalize(applied) == Refl(NatType(), numeral(5))


def test_add_zero_right_normalizes_multiple_inputs():
    lemma = add_n_0()

    for value in range(6):
        applied = App(lemma, numeral(value))
        assert normalize(applied) == Refl(NatType(), numeral(value))


def test_add_zero_right_applied_term_typechecks():
    lemma = add_n_0()
    three = numeral(3)
    applied = App(lemma, three)
    expected = Id(
        NatType(),
        App(App(add(), three), Zero()),
        three,
    )

    assert type_check(applied, expected)
