from mltt.common.span import Span
from mltt.elab.state import ElabState
from mltt.kernel.ast import App, Lam, Pi, Univ, Var
from mltt.kernel.env import Env


def test_spine_meta_solve() -> None:
    env = Env.of(Var(0), Univ(0))
    state = ElabState()
    meta = state.fresh_meta(env, Pi(Var(1), Var(1)), Span(0, 0), kind="implicit")
    state.add_constraint(env, App(meta, Var(0)), Var(0), Span(0, 0))
    state.solve(env)
    solution = state.metas[meta.mid].solution
    assert solution == Lam(Var(1), Var(0))
