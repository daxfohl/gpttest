from mltt.core.ast import App, Lam, Univ, Var
from mltt.core.reduce.beta import beta_step
from mltt.inductive.nat import Succ, Zero


def test_beta_step_reduces_single_application() -> None:
    term = App(Lam(Univ(), Succ(Var(0))), Zero())
    assert beta_step(term) == Succ(Zero())


def test_beta_step_progresses_once() -> None:
    term = App(Lam(Univ(), Succ(Var(0))), Succ(Zero()))
    step = beta_step(term)
    assert step == Succ(Succ(Zero()))
