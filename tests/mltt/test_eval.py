from mltt.ast import (
    App,
    IdElim,
    Lam,
    NatRec,
    NatType,
    Refl,
    Succ,
    TypeUniverse,
    Var,
    Zero,
)
from mltt.eval import beta_reduce, beta_step, normalize, whnf
from mltt.nat import add


def test_beta_reduce_performs_nested_reduction():
    inner_identity = Lam(TypeUniverse(), Var(0))
    term = App(Lam(TypeUniverse(), App(Var(0), Zero())), inner_identity)
    assert beta_reduce(term) == Zero()


def test_beta_reduce_unfolds_add_base_case():
    expected = Lam(
        NatType(),
        NatRec(
            P=Lam(NatType(), NatType()),
            z=Zero(),
            s=Lam(NatType(), Lam(NatType(), Succ(Var(0)))),
            n=Var(0),
        ),
    )

    assert beta_reduce(App(add, Zero())) == expected


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


def test_normalize_reduces_after_normalizing_function():
    curried = Lam(TypeUniverse(), Lam(TypeUniverse(), Var(0)))
    term = App(App(curried, Zero()), Zero())

    assert normalize(term) == Zero()


def test_beta_reduce_eta_expansion_collapses():
    term = App(Lam(TypeUniverse(), Var(0)), Lam(TypeUniverse(), Var(0)))
    assert beta_reduce(term) == Lam(TypeUniverse(), Var(0))


def test_whnf_stops_on_irreducible_function():
    term = App(Var(0), Zero())
    assert whnf(term) == App(Var(0), Zero())


def test_beta_step_progresses_once():
    term = App(Lam(TypeUniverse(), Succ(Var(0))), Succ(Zero()))
    step = beta_step(term)
    assert step == Succ(Succ(Zero()))


def test_normalize_complex_natrec():
    P = Lam(NatType(), NatType())
    z = Zero()
    s = Lam(NatType(), Lam(NatType(), Succ(Var(0))))
    term = NatRec(P, z, s, Succ(Succ(Zero())))
    assert normalize(term) == Succ(Succ(Zero()))
