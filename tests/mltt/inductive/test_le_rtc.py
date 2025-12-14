from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.le import LeRTCRefl, LeRTCStep, LeRTCType
from mltt.inductive.nat import Succ, Zero, numeral


def test_le_rtc_refl_typechecks() -> None:
    n = numeral(3)
    proof = LeRTCRefl(n)
    expected = LeRTCType(n, n)
    assert type_check(proof, expected)
    assert type_equal(infer_type(proof), expected)


def test_le_rtc_step_chain() -> None:
    n = Zero()
    proof = LeRTCRefl(n)
    proof = LeRTCStep(n, Zero(), proof)
    proof = LeRTCStep(n, Succ(Zero()), proof)
    expected = LeRTCType(n, numeral(2))

    assert type_check(proof, expected)
    assert type_equal(infer_type(proof), expected)
