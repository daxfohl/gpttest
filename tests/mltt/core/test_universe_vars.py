import pytest

from mltt.core.ast import ConstLevel, LevelVar, MaxOfLevels, SuccLevel, Univ
from mltt.core.debruijn import UCtx
from mltt.core.pretty import pretty


def test_level_var_shift_and_subst() -> None:
    assert LevelVar(0).shift(1) == LevelVar(1)
    assert LevelVar(1).shift(1, cutoff=2) == LevelVar(1)
    assert LevelVar(2).shift(-1, cutoff=1) == LevelVar(1)
    assert LevelVar(0).subst(ConstLevel(2), j=0) == ConstLevel(2)
    assert LevelVar(2).subst(ConstLevel(0), j=0) == LevelVar(1)


def test_level_var_instantiate() -> None:
    level = MaxOfLevels((LevelVar(1), SuccLevel(LevelVar(0))))
    actuals = [ConstLevel(0), ConstLevel(2)]
    assert level.instantiate(actuals) == MaxOfLevels(
        (ConstLevel(0), SuccLevel(ConstLevel(2)))
    )


def test_term_level_instantiate() -> None:
    term = Univ(MaxOfLevels((LevelVar(1), LevelVar(0))))
    actuals = [ConstLevel(1), ConstLevel(3)]
    assert term.level_instantiate(actuals) == Univ(
        MaxOfLevels((ConstLevel(1), ConstLevel(3)))
    )


def test_universe_var_scope_and_pretty() -> None:
    with pytest.raises(TypeError, match="Unbound universe variable"):
        Univ(LevelVar(0)).infer_type(uctx=UCtx.empty())

    uctx = UCtx.empty().push()
    assert Univ(LevelVar(0)).infer_type(uctx=uctx) == Univ(SuccLevel(LevelVar(0)))
    assert pretty(Univ(LevelVar(0))) == "Type(u0)"
