from mltt.core.ast import Lam, Pi, Var
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.eq import Id, Refl
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.top_bottom import BotRec, BotType, TopRec, TopType, Tt


def test_top_has_canonical_inhabitant() -> None:
    type_check(Tt(), TopType())


def test_toprec_eliminates_to_motive() -> None:
    motive = Lam(TopType(), NatType())
    case = Zero()
    term = TopRec(motive, case, Tt())

    type_check(term, NatType())
    assert type_equal(infer_type(term), NatType())


def test_botrec_ex_falso() -> None:
    motive = Lam(BotType(), NatType())
    lam = Lam(BotType(), BotRec(motive, Var(0)))
    expected_ty = Pi(BotType(), NatType())

    type_check(lam, expected_ty)
    assert type_equal(infer_type(lam), expected_ty)


def test_toprec_dependent_motive() -> None:
    motive = Lam(TopType(), Id(TopType(), Var(0), Var(0)))
    case = Refl(TopType(), Tt())
    term = TopRec(motive, case, Tt())
    expected = Id(TopType(), Tt(), Tt())

    type_check(term, expected)
    assert type_equal(infer_type(term), expected)
