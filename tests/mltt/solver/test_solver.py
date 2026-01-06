import pytest

from mltt.common.span import Span
from mltt.elab.errors import ElabError
from mltt.kernel.ast import App, MetaVar, Univ, Var
from mltt.kernel.env import Env
from mltt.kernel.levels import LConst, LMeta
from mltt.solver.solver import Solver


def test_occurs_check_fails() -> None:
    env = Env.of(Univ(0))
    solver = Solver()
    meta = solver.fresh_meta(env, Univ(0), Span(0, 0), kind="hole")
    with pytest.raises(ElabError, match="occurs check failed"):
        solver.add_constraint(env, meta, App(meta, Var(0)), Span(0, 0))


def test_level_meta_solves_to_lower_bound() -> None:
    solver = Solver()
    env = Env()
    level = solver.fresh_level_meta("type", Span(0, 0))
    assert isinstance(level, LMeta)
    solver.add_level_constraint(LConst(0), level, Span(0, 0))
    solver.solve(env)
    info = solver.level_metas[level.mid]
    assert info.solution == LConst(0)


def test_zonk_substitutes_meta_solution() -> None:
    env = Env.of(Univ(0))
    solver = Solver()
    meta = solver.fresh_meta(env, Univ(0), Span(0, 0), kind="implicit")
    solver.add_constraint(env, meta, Var(0), Span(0, 0))
    solver.solve(env)
    assert solver.zonk(MetaVar(meta.mid)) == Var(0)


def test_spine_meta_solve() -> None:
    env = Env.of(Var(0), Univ(0))
    solver = Solver()
    meta = solver.fresh_meta(env, Pi(Var(1), Var(1)), Span(0, 0), kind="implicit")
    solver.add_constraint(env, App(meta, Var(0)), Var(0), Span(0, 0))
    solver.solve(env)
    solution = solver.metas[meta.mid].solution
    assert solution == Lam(Var(1), Var(0))
