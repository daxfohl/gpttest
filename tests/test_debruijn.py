import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mltt.ast import App, Lam, Succ, TypeUniverse, Var
from mltt.debruijn import shift, subst


def test_shift_respects_cutoff():
    term = App(Var(1), Var(0))
    shifted = shift(term, by=2, cutoff=1)
    assert shifted == App(Var(3), Var(0))


def test_shift_through_lambda_increments_free_variable():
    term = Lam(TypeUniverse(), App(Var(1), Var(0)))
    shifted = shift(term, by=1, cutoff=0)
    assert shifted == Lam(TypeUniverse(), App(Var(2), Var(0)))


def test_subst_replaces_target_and_decrements_greater_indices():
    term = App(Var(1), Var(0))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert result == App(Var(0), Succ(Var(0)))


def test_subst_under_lambda_preserves_bound_variable():
    term = Lam(TypeUniverse(), App(Var(1), Var(0)))
    sub = Succ(Var(0))
    result = subst(term, sub)
    assert result == Lam(TypeUniverse(), App(Succ(Var(1)), Var(0)))


def test_subst_avoids_capture_under_nested_binder():
    term = Lam(TypeUniverse(), Var(1))
    sub = Var(0)
    result = subst(term, sub)
    assert result == Lam(TypeUniverse(), Var(1))
