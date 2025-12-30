import pytest

from mltt.kernel.ast import Pi, Univ
from mltt.kernel.levels import LevelVar, level_max, level_succ


def test_univ_level_succ_inference() -> None:
    u = LevelVar("u")
    assert Univ(u).infer_type() == Univ(level_succ(u))


def test_pi_level_uses_max() -> None:
    u = LevelVar("u")
    v = LevelVar("v")
    assert Pi(Univ(u), Univ(v)).infer_type() == Univ(
        level_max(level_succ(u), level_succ(v))
    )


def test_universe_cumulativity_with_vars() -> None:
    u = LevelVar("u")
    Univ(u).type_check(Univ(level_succ(u)))
    with pytest.raises(TypeError, match="Universe level mismatch"):
        Univ(level_succ(u)).type_check(Univ(u))
