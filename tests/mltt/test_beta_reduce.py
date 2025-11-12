from mltt.ast import (
    App,
    IdElim,
    Lam,
    NatRec,
    NatType,
    Refl,
    Succ,
    Univ,
    Var,
    Zero,
)
from mltt.beta_reduce import beta_reduce, beta_step, normalize, whnf
from mltt.nat import add


def test_beta_reduce_performs_nested_reduction():
    inner_identity = Lam(Univ(), Var(0))
    term = App(Lam(Univ(), App(Var(0), Zero())), inner_identity)
    assert beta_reduce(term) == Zero()


def test_beta_reduce_unfolds_add_base_case():
    expected = Lam(
        NatType(),
        NatRec(
            P=Lam(NatType(), NatType()),
            base=Var(0),
            step=Lam(NatType(), Lam(NatType(), Succ(Var(0)))),
            n=Zero(),
        ),
    )

    assert beta_reduce(App(add(), Zero())) == expected


def test_whnf_unfolds_natrec_on_successor():
    P = Lam(Univ(), Univ())
    z = Zero()
    s = Lam(Univ(), Lam(Univ(), Succ(Var(0))))
    term = NatRec(P, z, s, Succ(Zero()))

    result = whnf(term)

    assert result == App(App(s, Zero()), NatRec(P, z, s, Zero()))


def test_whnf_simplifies_identity_elimination_on_refl():
    term = IdElim(Var(0), Var(1), Var(2), Var(3), Var(4), Refl(Var(5), Var(6)))
    assert whnf(term) == Var(3)


def test_beta_step_reduces_single_application():
    term = App(Lam(Univ(), Succ(Var(0))), Zero())
    assert beta_step(term) == Succ(Zero())


def test_normalize_fully_reduces_application_chain():
    inner = Lam(Univ(), Succ(Var(0)))
    term = App(Lam(Univ(), App(inner, Var(0))), Zero())
    assert normalize(term) == Succ(Zero())


def test_normalize_reduces_after_normalizing_function():
    curried = Lam(Univ(), Lam(Univ(), Var(0)))
    term = App(App(curried, Zero()), Zero())

    assert normalize(term) == Zero()


def test_beta_reduce_eta_expansion_collapses():
    term = App(Lam(Univ(), Var(0)), Lam(Univ(), Var(0)))
    assert beta_reduce(term) == Lam(Univ(), Var(0))


def test_whnf_stops_on_irreducible_function():
    term = App(Var(0), Zero())
    assert whnf(term) == App(Var(0), Zero())


def test_beta_step_progresses_once():
    term = App(Lam(Univ(), Succ(Var(0))), Succ(Zero()))
    step = beta_step(term)
    assert step == Succ(Succ(Zero()))


def test_normalize_complex_natrec():
    P = Lam(NatType(), NatType())
    z = Zero()
    s = Lam(NatType(), Lam(NatType(), Succ(Var(0))))
    term = NatRec(P, z, s, Succ(Succ(Zero())))
    assert normalize(term) == Succ(Succ(Zero()))
