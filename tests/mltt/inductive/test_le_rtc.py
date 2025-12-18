from mltt.inductive.le import LeRTCRefl, LeRTCStep, LeRTCType
from mltt.inductive.nat import Succ, Zero, numeral


def test_le_rtc_refl_typechecks() -> None:
    n = numeral(3)
    proof = LeRTCRefl(n)
    expected = LeRTCType(n, n)
    proof.type_check(expected)
    assert proof.infer_type().type_equal(expected)


def test_le_rtc_step_chain() -> None:
    n = Zero()
    proof = LeRTCRefl(n)
    proof = LeRTCStep(n, Zero(), proof)
    proof = LeRTCStep(n, Succ(Zero()), proof)
    expected = LeRTCType(n, numeral(2))

    proof.type_check(expected)
    assert proof.infer_type().type_equal(expected)
