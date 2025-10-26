from mltt.ast import App, Id, NatType, Pi, Refl, Succ, Var, Zero
from mltt.eval import normalize
from mltt.nat import add, add_zero_right, add_succ_right, numeral
from mltt.typing import type_check


def test_add_has_expected_pi_type():
    add_type = Pi(NatType(), Pi(NatType(), NatType()))

    assert type_check(add, add_type)


def test_add_zero_left_identity():
    n_term = numeral(4)
    expected = numeral(4)

    result = normalize(App(App(add, Zero()), n_term))

    assert result == expected


def test_add_satisfies_recursive_step():
    k_term = numeral(2)
    n_term = numeral(3)

    lhs = normalize(App(App(add, Succ(k_term)), n_term))
    rhs = normalize(Succ(App(App(add, k_term), n_term)))

    assert lhs == rhs


def test_add_produces_expected_numeral():
    result = normalize(App(App(add, numeral(2)), numeral(3)))

    assert result == numeral(5)


def test_add_zero_right_typechecks_and_reduces():
    lemma = add_zero_right()
    lemma_ty = Pi(
        NatType(),
        Id(NatType(), App(App(add, Var(0)), Zero()), Var(0)),
    )
    assert type_check(lemma, lemma_ty)
    assert normalize(App(lemma, numeral(4))) == Refl(NatType(), numeral(4))


def test_add_succ_right_typechecks():
    lemma = add_succ_right()
    lemma_ty = Pi(
        NatType(),
        Pi(
            NatType(),
            Id(
                NatType(),
                App(App(add, Var(1)), Succ(Var(0))),
                Succ(App(App(add, Var(1)), Var(0))),
            ),
        ),
    )
    assert type_check(lemma, lemma_ty)


def test_add_succ_right_normalizes():
    lemma = add_succ_right()
    applied = App(App(lemma, numeral(2)), numeral(3))
    assert normalize(applied) == Refl(
        NatType(),
        Succ(normalize(App(App(add, numeral(2)), numeral(3)))),
    )
