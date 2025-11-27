from mltt.ast import App, Lam, Univ, Var
from mltt.eq import IdElim, Refl
from mltt.nat import NatRec, NatType, Succ, Zero, add
from mltt.normalization import beta_step, normalize, whnf, normalize_step


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
            step=Lam(NatType(), Lam(NatType(), Succ(Var(0)))),
            n=Zero(),
        ),
    )

    assert normalize_step(App(add(), Zero())) == expected


def test_whnf_unfolds_natrec_on_successor() -> None:
    P = Lam(Univ(), Univ())
    z = Zero()
    s = Lam(Univ(), Lam(Univ(), Succ(Var(0))))
    term = NatRec(P=P, base=z, step=s, n=Succ(Zero()))
    result = whnf(term)

    assert result == Succ(
        NatRec(
            P=Lam(ty=Univ(0), body=Univ(0)),
            base=Zero(),
            step=Lam(ty=Univ(0), body=Lam(ty=Univ(0), body=Succ(Var(0)))),
            n=Zero(),
        )
    )


def test_whnf_simplifies_identity_elimination_on_refl() -> None:
    term = IdElim(
        A=Var(0),
        x=Var(1),
        P=Var(2),
        d=Var(3),
        y=Var(4),
        p=Refl(Var(5), Var(6)),
    )
    assert whnf(term) == Var(3)


def test_beta_step_reduces_single_application() -> None:
    term = App(Lam(Univ(), Succ(Var(0))), Zero())
    assert beta_step(term) == Succ(Zero())


def test_normalize_fully_reduces_application_chain() -> None:
    inner = Lam(Univ(), Succ(Var(0)))
    term = App(Lam(Univ(), App(inner, Var(0))), Zero())
    assert normalize(term) == Succ(Zero())


def test_normalize_reduces_after_normalizing_function() -> None:
    curried = Lam(Univ(), Lam(Univ(), Var(0)))
    term = App(App(curried, Zero()), Zero())

    assert normalize(term) == Zero()


def test_normalize_eta_expansion_collapses() -> None:
    term = App(Lam(Univ(), Var(0)), Lam(Univ(), Var(0)))
    assert normalize(term) == Lam(Univ(), Var(0))


def test_whnf_stops_on_irreducible_function() -> None:
    term = App(Var(0), Zero())
    assert whnf(term) == App(Var(0), Zero())


def test_beta_step_progresses_once() -> None:
    term = App(Lam(Univ(), Succ(Var(0))), Succ(Zero()))
    step = beta_step(term)
    assert step == Succ(Succ(Zero()))


def test_normalize_complex_natrec() -> None:
    P = Lam(NatType(), NatType())
    z = Zero()
    s = Lam(NatType(), Lam(NatType(), Succ(Var(0))))
    term = NatRec(P=P, base=z, step=s, n=Succ(Succ(Zero())))
    assert normalize(term) == Succ(Succ(Zero()))
