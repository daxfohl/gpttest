from mltt.kernel.ast import App, Lam, Pi, Var
from mltt.kernel.debruijn import mk_lams
from mltt.kernel.pretty import pretty
from mltt.inductive.nat import NatType, Succ, Zero


def test_inductive_and_constructor_names() -> None:
    nat = NatType()
    assert pretty(nat) == "Nat"
    assert pretty(Zero()) == "Zero"
    assert pretty(Succ(Zero())) == "Succ Zero"


def test_lambda_and_pi_rendering() -> None:
    nat = NatType()
    assert pretty(Lam(nat, Var(0))) == "\\x : Nat. x"
    assert pretty(Pi(nat, nat)) == "Nat -> Nat"
    assert pretty(Pi(nat, Var(0))) == "Pi x : Nat. x"


def test_nested_application_uses_binder_names() -> None:
    nat = NatType()
    term = mk_lams(nat, nat, body=App(Var(1), Var(0)))
    assert pretty(term) == "\\x : Nat. \\x1 : Nat. x x1"
