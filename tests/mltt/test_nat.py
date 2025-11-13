from mltt.ast import App, Id, NatType, Pi, Refl, Succ, Term, Var, Zero
from mltt.beta_reduce import normalize
from mltt.nat import add, add_terms, add_n_0, numeral
from mltt.typing import type_check


def test_add_has_expected_pi_type() -> None:
    add_type = Pi(NatType(), Pi(NatType(), NatType()))

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


def test_add_zero_right_typechecks() -> None:
    lemma = add_n_0()
    lemma_ty = Pi(
        NatType(),
        Id(NatType(), add_terms(Var(0), Zero()), Var(0)),
    )

    assert type_check(lemma, lemma_ty)


def test_add_zero_right_normalizes() -> None:
    lemma = add_n_0()
    applied = App(lemma, numeral(5))
    assert normalize(applied) == Refl(NatType(), numeral(5))


def test_add_zero_right_normalizes_multiple_inputs() -> None:
    lemma = add_n_0()

    for value in range(6):
        applied = App(lemma, numeral(value))
        assert normalize(applied) == Refl(NatType(), numeral(value))


def test_add_zero_right_applied_term_typechecks() -> None:
    lemma = add_n_0()
    three = numeral(3)
    applied = App(lemma, three)
    expected = Id(
        NatType(),
        add_terms(three, Zero()),
        three,
    )

    assert type_check(applied, expected)
