import mltt.vec as vec
from mltt.ast import Pi, Univ
from mltt.nat import NatType, Succ, Zero
from mltt.typing import infer_type, type_check


def test_infer_vec_type() -> None:
    assert infer_type(vec.Vec) == Pi(Univ(0), Pi(NatType(), Univ(0)))


def test_nil_has_zero_length() -> None:
    elem_ty = NatType()
    nil = vec.Nil(elem_ty)
    assert type_check(nil, vec.VecType(elem_ty, Zero()))


def test_cons_increments_length() -> None:
    elem_ty = NatType()
    tail = vec.Nil(elem_ty)
    cons = vec.Cons(elem_ty, Zero(), Zero(), tail)
    assert type_check(cons, vec.VecType(elem_ty, Succ(Zero())))
