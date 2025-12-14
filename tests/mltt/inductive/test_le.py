from mltt.core.ast import Var
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.le import LeRefl, LeStep, LeType
from mltt.inductive.nat import NatType, Succ, Zero, numeral


def test_lerefl_typechecks() -> None:
    n = numeral(2)
    assert type_check(LeRefl(n), LeType(n, n))


def test_lestep_typechecks() -> None:
    n = Zero()
    m = Zero()
    p = LeRefl(m)
    proof = LeStep(n, m, p)
    assert type_check(proof, LeType(n, Succ(m)))


def test_lestep_chain_builds_longer_proof() -> None:
    proof = LeStep(Zero(), Zero(), LeRefl(Zero()))
    proof2 = LeStep(Zero(), Succ(Zero()), proof)
    assert type_check(proof2, LeType(Zero(), numeral(2)))


def test_infer_type_le_refl() -> None:
    n = numeral(3)
    assert type_equal(infer_type(LeRefl(n)), LeType(n, n))
