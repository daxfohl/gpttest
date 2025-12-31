from mltt.inductive.eq import Id, Refl
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.unit_empty import (
    EmptyElim,
    EmptyType,
    UnitElim,
    UnitType,
    UnitValue,
)
from mltt.kernel.ast import Lam, Pi, Var


def test_top_has_canonical_inhabitant() -> None:
    UnitValue().type_check(UnitType())


def test_toprec_eliminates_to_motive() -> None:
    motive = Lam(UnitType(), NatType())
    case = Zero()
    term = UnitElim(motive, case, UnitValue())

    term.type_check(NatType())
    assert term.infer_type().type_equal(NatType())


def test_botrec_ex_falso() -> None:
    motive = Lam(EmptyType(), NatType())
    lam = Lam(EmptyType(), EmptyElim(motive, Var(0)))
    expected_ty = Pi(EmptyType(), NatType())

    lam.type_check(expected_ty)
    assert lam.infer_type().type_equal(expected_ty)


def test_toprec_dependent_motive() -> None:
    motive = Lam(UnitType(), Id(UnitType(), Var(0), Var(0)))
    case = Refl(UnitType(), UnitValue())
    term = UnitElim(motive, case, UnitValue())
    expected = Id(UnitType(), UnitValue(), UnitValue())

    term.type_check(expected)
    assert term.infer_type().type_equal(expected)
