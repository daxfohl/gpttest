import pytest

from mltt.core.ast import Pi, Univ, Var
from mltt.core.inductive_utils import apply_term, nested_pi
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.bool import (
    BoolType,
    False_,
    FalseCtor,
    True_,
    TrueCtor,
    and_,
    and_terms,
    if_,
    if_terms,
    not_,
    not_term,
    or_,
    or_terms,
)
from mltt.inductive.nat import NatType, Succ, Zero, numeral


def test_ctor_types() -> None:
    assert infer_type(FalseCtor) == BoolType()
    assert infer_type(TrueCtor) == BoolType()


def test_not_normalizes() -> None:
    assert normalize(not_term(False_())) == True_()
    assert normalize(not_term(True_())) == False_()


def test_and_normalizes() -> None:
    assert normalize(and_terms(True_(), True_())) == True_()
    assert normalize(and_terms(True_(), False_())) == False_()
    assert normalize(and_terms(False_(), True_())) == False_()


def test_or_normalizes() -> None:
    assert normalize(or_terms(True_(), False_())) == True_()
    assert normalize(or_terms(False_(), True_())) == True_()
    assert normalize(or_terms(False_(), False_())) == False_()


def test_operator_types() -> None:
    bool_to_bool = nested_pi(BoolType(), return_ty=BoolType())
    bool_binop = nested_pi(BoolType(), BoolType(), return_ty=BoolType())

    assert type_check(not_(), bool_to_bool)
    assert type_check(and_(), bool_binop)
    assert type_check(or_(), bool_binop)


def test_and_rejects_non_bool_argument() -> None:
    with pytest.raises(TypeError):
        type_check(and_terms(True_(), numeral(0)), BoolType())


def test_if_type() -> None:
    expected = nested_pi(
        Univ(0),
        BoolType(),
        Var(1),
        Var(2),
        return_ty=Var(3),
    )

    assert type_equal(infer_type(if_()), expected)


def test_if_normalizes() -> None:
    nat_if_true = apply_term(if_(), NatType(), True_(), Zero(), Succ(Zero()))
    nat_if_false = if_terms(NatType(), False_(), Zero(), Succ(Zero()))

    assert normalize(nat_if_true) == Zero()
    assert normalize(nat_if_false) == Succ(Zero())
