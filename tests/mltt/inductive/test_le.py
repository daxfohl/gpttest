from mltt.inductive.le import LeRefl, LeStep, LeType
from mltt.inductive.nat import Succ, Zero, numeral


def test_lerefl_typechecks() -> None:
    n = numeral(2)

    LeRefl(n).type_check(LeType(n, n))


def test_lestep_typechecks() -> None:
    n = Zero()
    m = Zero()
    p = LeRefl(m)
    proof = LeStep(n, m, p)

    proof.type_check(LeType(n, Succ(m)))


def test_lestep_chain_builds_longer_proof() -> None:
    proof = LeStep(Zero(), Zero(), LeRefl(Zero()))
    proof2 = LeStep(Zero(), Succ(Zero()), proof)

    proof2.type_check(LeType(Zero(), numeral(2)))


def test_infer_type_le_refl() -> None:
    n = numeral(3)
    assert LeRefl(n).infer_type().type_equal(LeType(n, n))
