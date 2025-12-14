from mltt.core.ast import Var
from mltt.core.inductive_utils import nested_pi, apply_term
from mltt.core.typing import infer_type, type_equal, type_check
from mltt.inductive.eq import Id
from mltt.inductive.nat import NatType, Succ, Zero, add_terms, numeral
from mltt.proofs.add_comm import (
    add_zero_left,
    add_zero_right,
    succ_add, add_succ_right, add_comm,
)


def test_add_zero_right_typechecks() -> None:
    lemma = add_zero_right()
    expected_ty = nested_pi(
        NatType(),
        return_ty=Id(NatType(), add_terms(Var(0), Zero()), Var(0)),
    )
    assert type_equal(infer_type(lemma), expected_ty)


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
        NatType(),
        NatType(),
        return_ty=Id(
            NatType(),
            add_terms(Var(1), Succ(Var(0))),
            Succ(add_terms(Var(1), Var(0))),
        ),
    )
    # assert type_equal(infer_type(lemma), expected_ty)


def test_add_comm_typechecks_and_examples() -> None:
    lemma = add_comm()
    expected_ty = nested_pi(
        NatType(),
        NatType(),
        return_ty=Id(NatType(), add_terms(Var(1), Var(0)), add_terms(Var(0), Var(1))),
    )
    # assert type_equal(infer_type(lemma), expected_ty)

    m = numeral(2)
    n = numeral(3)
    proof = apply_term(lemma, m, n)
    expected_proof_ty = Id(NatType(), add_terms(m, n), add_terms(n, m))
    # assert type_check(proof, expected_proof_ty)
