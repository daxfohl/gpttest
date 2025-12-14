from mltt.core.ast import Lam, Var
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.rtc import RTCRefl, RTCStep, RTCType


def nat_relation() -> Lam:
    # A simple relation returning Nat to keep typing straightforward.
    return Lam(NatType(), Lam(NatType(), NatType()))


def test_rtc_refl_typechecks() -> None:
    A = NatType()
    R = nat_relation()
    proof = RTCRefl(A, R, Zero())
    expected = RTCType(A, R, Zero(), Zero())
    assert type_check(proof, expected)
    assert type_equal(infer_type(proof), expected)


def test_rtc_step_typechecks() -> None:
    A = NatType()
    R = nat_relation()
    x = Zero()
    y = Zero()
    z = Zero()
    step = Zero()
    ih = RTCRefl(A, R, z)
    proof = RTCStep(A, R, x, z, y, step, ih)
    expected = RTCType(A, R, x, z)
    assert type_check(proof, expected)
    assert type_equal(infer_type(proof), expected)
