from mltt.core.ast import Var
from mltt.core.inductive_utils import nested_pi
from mltt.core.typing import infer_type, type_equal
from mltt.inductive.eq import Id
from mltt.inductive.nat import NatType, Succ, Zero, add_terms
from mltt.proofs.add_comm import (
    add_zero_left,
    add_zero_right,
    succ_add,
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
