from mltt.core.ast import App, Lam, Univ, Var
from mltt.core.reduce.normalize import normalize, normalize_step
from mltt.inductive.nat import NatRec, NatType, Succ, Zero, add


def test_normalize_performs_nested_reduction() -> None:
    inner_identity = Lam(Univ(), Var(0))
    a = Var(0)
    b = Zero()
    a1 = Lam(Univ(), App(a, b))
    term = App(a1, inner_identity)
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

    a = add()
    b = Zero()
    assert normalize_step(App(a, b)) == expected


def test_normalize_fully_reduces_application_chain() -> None:
    inner = Lam(Univ(), Succ(Var(0)))
    b = Var(0)
    a = Lam(Univ(), App(inner, b))
    b1 = Zero()
    term = App(a, b1)
    assert normalize(term) == Succ(Zero())


def test_normalize_reduces_after_normalizing_function() -> None:
    curried = Lam(Univ(), Lam(Univ(), Var(0)))
    b = Zero()
    a = App(curried, b)
    b1 = Zero()
    term = App(a, b1)

    assert normalize(term) == Zero()


def test_normalize_eta_expansion_collapses() -> None:
    a = Lam(Univ(), Var(0))
    b = Lam(Univ(), Var(0))
    term = App(a, b)
    assert normalize(term) == Lam(Univ(), Var(0))


def test_normalize_complex_natrec() -> None:
    P = Lam(NatType(), NatType())
    z = Zero()
    s = Lam(NatType(), Lam(NatType(), Succ(Var(0))))
    term = NatRec(P=P, base=z, step=s, n=Succ(Succ(Zero())))
    assert normalize(term) == Succ(Succ(Zero()))
