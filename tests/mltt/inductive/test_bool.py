import pytest

from mltt.core.ast import Univ, Var
from mltt.core.reduce.normalize import normalize
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.core.util import apply_term, nested_pi
from mltt.inductive.bool import (
    BoolType,
    False_,
    FalseCtor,
    True_,
    TrueCtor,
    and_term,
    and_,
    if_term,
    if_,
    not_term,
    not_,
    or_term,
    or_,
)
from mltt.inductive.nat import NatType, Succ, Zero, numeral


def test_ctor_types() -> None:
    assert infer_type(FalseCtor) == BoolType()
    assert infer_type(TrueCtor) == BoolType()


def test_not_normalizes() -> None:
    assert normalize(not_(False_())) == True_()
    assert normalize(not_(True_())) == False_()


def test_and_normalizes() -> None:
    assert normalize(and_(True_(), True_())) == True_()
    assert normalize(and_(True_(), False_())) == False_()
    assert normalize(and_(False_(), True_())) == False_()


def test_or_normalizes() -> None:
    assert normalize(or_(True_(), False_())) == True_()
    assert normalize(or_(False_(), True_())) == True_()
    assert normalize(or_(False_(), False_())) == False_()


def test_operator_types() -> None:
    bool_to_bool = nested_pi(BoolType(), return_ty=BoolType())
    bool_binop = nested_pi(BoolType(), BoolType(), return_ty=BoolType())

    type_check(not_term(), bool_to_bool)
    type_check(and_term(), bool_binop)
    type_check(or_term(), bool_binop)


def test_and_rejects_non_bool_argument() -> None:
    with pytest.raises(TypeError):
        type_check(and_(True_(), numeral(0)), BoolType())


def test_if_type() -> None:
    expected = nested_pi(
        Univ(0),
        BoolType(),
        Var(1),
        Var(2),
        return_ty=Var(3),
    )

    assert type_equal(infer_type(if_term()), expected)


def test_if_normalizes() -> None:
    nat_if_true = apply_term(if_term(), NatType(), True_(), Zero(), Succ(Zero()))
    nat_if_false = if_(NatType(), False_(), Zero(), Succ(Zero()))

    assert normalize(nat_if_true) == Zero()
    assert normalize(nat_if_false) == Succ(Zero())
