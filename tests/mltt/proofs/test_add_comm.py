from mltt.core.ast import Var
from mltt.core.inductive_utils import nested_pi, apply_term
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_equal, type_check
from mltt.inductive.eq import Id, Refl
from mltt.inductive.nat import NatType, Succ, Zero, add_terms, numeral
from mltt.proofs.add_comm import (
    add_zero_left,
    add_zero_right,
    succ_add,
    add_succ_right,
    add_comm,
)


def test_add_zero_right_typechecks() -> None:
    lemma = add_zero_right()
    expected_ty = nested_pi(
        NatType(),
        return_ty=Id(NatType(), add_terms(Var(0), Zero()), Var(0)),
    )
    assert type_equal(infer_type(lemma), expected_ty)


def test_add_zero_right_normalizes() -> None:
    lemma = add_zero_right()
    applied = apply_term(lemma, numeral(5))
    assert normalize(applied) == Refl(NatType(), numeral(5))


def test_add_zero_right_normalizes_multiple_inputs() -> None:
    lemma = add_zero_right()

    for value in range(6):
        applied = apply_term(lemma, numeral(value))
        assert normalize(applied) == Refl(NatType(), numeral(value))


def test_add_zero_right_applied_term_typechecks() -> None:
    lemma = add_zero_right()
    three = numeral(3)
    applied = apply_term(lemma, three)
    expected = Id(
        NatType(),
        add_terms(three, Zero()),
        three,
    )

    assert type_check(applied, expected)


def test_add_zero_left_typechecks() -> None:
    lemma = add_zero_left()
    expected_ty = nested_pi(
        NatType(),
        return_ty=Id(NatType(), add_terms(Zero(), Var(0)), Var(0)),
    )
    assert type_equal(infer_type(lemma), expected_ty)


def test_succ_add_typechecks() -> None:
    lemma = succ_add()
    expected_ty = nested_pi(
        NatType(),
        NatType(),
        return_ty=Id(
            NatType(),
            add_terms(Succ(Var(1)), Var(0)),
            Succ(add_terms(Var(1), Var(0))),
        ),
    )
    assert type_equal(infer_type(lemma), expected_ty)


def test_add_succ_right_typechecks() -> None:
    lemma = add_succ_right()
    expected_ty = nested_pi(
        NatType(),  # n
        NatType(),  # m
        return_ty=Id(
            NatType(),
            # this is add m (Succ n) in your definitional encoding:
            add_terms(Var(0), Succ(Var(1))),  # note the swap
            Succ(add_terms(Var(0), Var(1))),
        ),
    )
    assert type_equal(infer_type(lemma), expected_ty)


def test_add_comm_typechecks_and_examples() -> None:
    lemma = add_comm()
    expected_ty = nested_pi(
        NatType(),
        NatType(),
        return_ty=Id(NatType(), add_terms(Var(1), Var(0)), add_terms(Var(0), Var(1))),
    )
    assert type_equal(infer_type(lemma), expected_ty)

    m = numeral(2)
    n = numeral(3)
    proof = apply_term(lemma, m, n)
    expected_proof_ty = Id(NatType(), add_terms(m, n), add_terms(n, m))
    assert type_check(proof, expected_proof_ty)
