from mltt.inductive.nat import NatRec, NatType, Succ, Zero, add_term
from mltt.kernel.ast import App, Lam, Univ, Var
from mltt.kernel.telescope import mk_lams


def test_normalize_performs_nested_reduction() -> None:
    inner_identity = Lam(Univ(), Var(0))
    term = App(Lam(Univ(), App(Var(0), Zero())), inner_identity)
    assert term.normalize() == Zero()


def test_normalize_step_unfolds_add_base_case() -> None:
    expected = Lam(
        NatType(),
        NatRec(
            A=NatType(),
            base=Var(0),
            step=mk_lams(NatType(), NatType(), body=Succ(Var(0))),
            n=Zero(),
        ),
    )

    assert App(add_term(), Zero()).normalize_step() == expected


def test_normalize_fully_reduces_application_chain() -> None:
    inner = Lam(Univ(), Succ(Var(0)))
    term = App(Lam(Univ(), App(inner, Var(0))), Zero())
    assert term.normalize() == Succ(Zero())


def test_normalize_reduces_after_normalizing_function() -> None:
    curried = mk_lams(Univ(), Univ(), body=Var(0))
    term = App(App(curried, Zero()), Zero())

    assert term.normalize() == Zero()


def test_normalize_eta_expansion_collapses() -> None:
    term = App(Lam(Univ(), Var(0)), Lam(Univ(), Var(0)))
    assert term.normalize() == Lam(Univ(), Var(0))


def test_normalize_complex_natrec() -> None:
    A = NatType()
    z = Zero()
    s = mk_lams(NatType(), NatType(), body=Succ(Var(0)))
    term = NatRec(A=A, base=z, step=s, n=Succ(Succ(Zero())))
    assert term.normalize() == Succ(Succ(Zero()))
