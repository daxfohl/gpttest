import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mltt.ast import (
    App,
    IdElim,
    Lam,
    NatRec,
    Refl,
    Succ,
    TypeUniverse,
    Var,
    Zero,
)
from mltt.eval import beta_reduce, beta_step, normalize, whnf


def test_beta_reduce_performs_nested_reduction():
    inner_identity = Lam(TypeUniverse(), Var(0))
    term = App(Lam(TypeUniverse(), App(Var(0), Zero())), inner_identity)
    assert beta_reduce(term) == Zero()


def test_whnf_unfolds_natrec_on_successor():
    P = Lam(TypeUniverse(), TypeUniverse())
    z = Zero()
    s = Lam(TypeUniverse(), Lam(TypeUniverse(), Succ(Var(0))))
    term = NatRec(P, z, s, Succ(Zero()))

    result = whnf(term)

    assert result == App(App(s, Zero()), NatRec(P, z, s, Zero()))


def test_whnf_simplifies_identity_elimination_on_refl():
    term = IdElim(Var(0), Var(1), Var(2), Var(3), Var(4), Refl(Var(5), Var(6)))
    assert whnf(term) == Var(3)


def test_beta_step_reduces_single_application():
    term = App(Lam(TypeUniverse(), Succ(Var(0))), Zero())
    assert beta_step(term) == Succ(Zero())


def test_normalize_fully_reduces_application_chain():
    inner = Lam(TypeUniverse(), Succ(Var(0)))
    term = App(Lam(TypeUniverse(), App(inner, Var(0))), Zero())
    assert normalize(term) == Succ(Zero())
