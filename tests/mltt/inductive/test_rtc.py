from mltt.core.ast import Lam, Var
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.eq import Id, Refl
from mltt.inductive.nat import NatType, Succ, Zero
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


def succ_relation() -> Lam:
    return Lam(
        NatType(),
        Lam(NatType(), Id(NatType(), Succ(Var(1)), Var(0))),
    )


def test_rtc_succ_chain_two_steps() -> None:
    A = NatType()
    R = succ_relation()
    one = Succ(Zero())
    two = Succ(one)

    # Path 1 -> 2
    base = RTCRefl(A, R, two)
    step1 = RTCStep(A, R, one, two, two, Refl(NatType(), two), base)

    # Path 0 -> 2 by prepending the 0 -> 1 edge
    proof = RTCStep(A, R, Zero(), two, one, Refl(NatType(), one), step1)
    expected = RTCType(A, R, Zero(), two)
    assert type_check(proof, expected)
    assert type_equal(infer_type(proof), expected)
