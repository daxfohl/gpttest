from mltt.core.ast import Lam, Pi, Var
from mltt.core.inductive_utils import apply_term
from mltt.core.typing import infer_type, type_check, type_equal
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.top_bottom import BotRec, BotType, TopRec, TopType, Tt


def test_top_has_canonical_inhabitant() -> None:
    assert type_check(Tt(), TopType())


def test_toprec_eliminates_to_motive() -> None:
    motive = Lam(TopType(), NatType())
    case = Zero()
    term = TopRec(motive, case, Tt())

    assert type_check(term, NatType())
    assert type_equal(infer_type(term), NatType())


def test_botrec_ex_falso() -> None:
    motive = Lam(BotType(), NatType())
    lam = Lam(BotType(), BotRec(motive, Var(0)))
    expected_ty = Pi(BotType(), NatType())

    assert type_check(lam, expected_ty)
    assert type_equal(infer_type(lam), expected_ty)
