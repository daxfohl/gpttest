import pytest

from mltt.core.ast import Univ, Var
from mltt.core.debruijn import mk_app, mk_pis
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
    assert FalseCtor.infer_type() == BoolType()
    assert TrueCtor.infer_type() == BoolType()


def test_not_normalizes() -> None:
    assert not_(False_()).normalize() == True_()
    assert not_(True_()).normalize() == False_()


def test_and_normalizes() -> None:
    assert and_(True_(), True_()).normalize() == True_()
    assert and_(True_(), False_()).normalize() == False_()
    assert and_(False_(), True_()).normalize() == False_()


def test_or_normalizes() -> None:
    assert or_(True_(), False_()).normalize() == True_()
    assert or_(False_(), True_()).normalize() == True_()
    assert or_(False_(), False_()).normalize() == False_()


def test_operator_types() -> None:
    bool_to_bool = mk_pis(BoolType(), return_ty=BoolType())
    bool_binop = mk_pis(BoolType(), BoolType(), return_ty=BoolType())

    not_term().type_check(bool_to_bool)
    and_term().type_check(bool_binop)
    or_term().type_check(bool_binop)


def test_and_rejects_non_bool_argument() -> None:
    with pytest.raises(TypeError):
        and_(True_(), numeral(0)).type_check(BoolType())


def test_if_type() -> None:
    expected = mk_pis(
        Univ(0),
        BoolType(),
        Var(1),
        Var(2),
        return_ty=Var(3),
    )

    assert if_term().infer_type().type_equal(expected)


def test_if_normalizes() -> None:
    nat_if_true = mk_app(if_term(), NatType(), True_(), Zero(), Succ(Zero()))
    nat_if_false = if_(NatType(), False_(), Zero(), Succ(Zero()))

    assert nat_if_true.normalize() == Zero()
    assert nat_if_false.normalize() == Succ(Zero())
