from mltt.core.ast import App, IdElim, Lam, Refl, Var
from mltt.core.inductive_utils import nested_lam
from mltt.core.reduce.iota import iota_head_step
from mltt.inductive.nat import NatRec, NatType, Succ, Zero


def test_idelim_on_refl_returns_d() -> None:
    term = IdElim(
        A=Var(0),
        x=Var(1),
        P=Var(2),
        d=Var(3),
        y=Var(4),
        p=Refl(Var(5), Var(6)),
    )
    assert iota_head_step(term) == Var(3)


def test_natrec_on_zero_reduces_to_base() -> None:
    P = Lam(NatType(), NatType())
    base = Succ(Zero())
    step = nested_lam(NatType(), NatType(), body=Var(0))
    rec = NatRec(P=P, base=base, step=step, n=Zero())
    assert iota_head_step(rec) == base


def test_natrec_on_succ_expands_step_and_ih() -> None:
    P = Lam(NatType(), NatType())
    base = Zero()
    step = nested_lam(NatType(), NatType(), body=Var(0))
    rec = NatRec(P=P, base=base, step=step, n=Succ(Zero()))
    result = iota_head_step(rec)
    expected = App(App(step, Zero()), NatRec(P=P, base=base, step=step, n=Zero()))
    assert result == expected
