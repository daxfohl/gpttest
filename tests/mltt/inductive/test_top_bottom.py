from mltt.inductive.eq import Id, Refl
from mltt.inductive.nat import NatType, Zero
from mltt.inductive.top_bottom import BotElim, BotType, TopElim, TopType, Tt
from mltt.kernel.ast import Lam, Pi, Var


def test_top_has_canonical_inhabitant() -> None:
    Tt().type_check(TopType())


def test_toprec_eliminates_to_motive() -> None:
    motive = Lam(TopType(), NatType())
    case = Zero()
    term = TopElim(motive, case, Tt())

    term.type_check(NatType())
    assert term.infer_type().type_equal(NatType())


def test_botrec_ex_falso() -> None:
    motive = Lam(BotType(), NatType())
    lam = Lam(BotType(), BotElim(motive, Var(0)))
    expected_ty = Pi(BotType(), NatType())

    lam.type_check(expected_ty)
    assert lam.infer_type().type_equal(expected_ty)


def test_toprec_dependent_motive() -> None:
    motive = Lam(TopType(), Id(TopType(), Var(0), Var(0)))
    case = Refl(TopType(), Tt())
    term = TopElim(motive, case, Tt())
    expected = Id(TopType(), Tt(), Tt())

    term.type_check(expected)
    assert term.infer_type().type_equal(expected)
