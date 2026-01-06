import pytest

from mltt.common.span import Span
from mltt.elab.errors import ElabError
from mltt.kernel.ast import App, MetaVar, Univ, Var
from mltt.kernel.env import Env
from mltt.kernel.levels import LConst, LMeta
from mltt.solver.solver import Solver


def test_occurs_check_fails() -> None:
    env = Env.of(Univ(0))
    state = Solver()
    meta = state.fresh_meta(env, Univ(0), Span(0, 0), kind="hole")
    with pytest.raises(ElabError, match="occurs check failed"):
        state.add_constraint(env, meta, App(meta, Var(0)), Span(0, 0))


def test_level_meta_solves_to_lower_bound() -> None:
    state = Solver()
    env = Env()
    level = state.fresh_level_meta("type", Span(0, 0))
    assert isinstance(level, LMeta)
    state.add_level_constraint(LConst(0), level, Span(0, 0))
    state.solve(env)
    info = state.level_metas[level.mid]
    assert info.solution == LConst(0)


def test_zonk_substitutes_meta_solution() -> None:
    env = Env.of(Univ(0))
    state = Solver()
    meta = state.fresh_meta(env, Univ(0), Span(0, 0), kind="implicit")
    state.add_constraint(env, meta, Var(0), Span(0, 0))
    state.solve(env)
    assert state.zonk(MetaVar(meta.mid)) == Var(0)
