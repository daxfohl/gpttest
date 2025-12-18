from mltt.core.ast import App, Lam, Univ, Var
from mltt.core.util import nested_lam
from mltt.core.reduce.normalize import normalize, normalize_step
from mltt.inductive.nat import NatRec, NatType, Succ, Zero, add_term


def test_normalize_performs_nested_reduction() -> None:
    inner_identity = Lam(Univ(), Var(0))
    term = App(Lam(Univ(), App(Var(0), Zero())), inner_identity)
    assert normalize(term) == Zero()


def test_normalize_step_unfolds_add_base_case() -> None:
    expected = Lam(
        NatType(),
        NatRec(
            P=Lam(NatType(), NatType()),
            base=Var(0),
            step=nested_lam(NatType(), NatType(), body=Succ(Var(0))),
            n=Zero(),
        ),
    )

    assert normalize_step(App(add_term(), Zero())) == expected


def test_normalize_fully_reduces_application_chain() -> None:
    inner = Lam(Univ(), Succ(Var(0)))
    term = App(Lam(Univ(), App(inner, Var(0))), Zero())
    assert normalize(term) == Succ(Zero())


def test_normalize_reduces_after_normalizing_function() -> None:
    curried = nested_lam(Univ(), Univ(), body=Var(0))
    term = App(App(curried, Zero()), Zero())

    assert normalize(term) == Zero()


def test_normalize_eta_expansion_collapses() -> None:
    term = App(Lam(Univ(), Var(0)), Lam(Univ(), Var(0)))
    assert normalize(term) == Lam(Univ(), Var(0))


def test_normalize_complex_natrec() -> None:
    P = Lam(NatType(), NatType())
    z = Zero()
    s = nested_lam(NatType(), NatType(), body=Succ(Var(0)))
    term = NatRec(P=P, base=z, step=s, n=Succ(Succ(Zero())))
    assert normalize(term) == Succ(Succ(Zero()))
