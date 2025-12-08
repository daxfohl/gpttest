from mltt.core.ast import App, IdElim, Lam, Refl, Univ, Var
from mltt.core.reduce.whnf import whnf
from mltt.inductive.nat import NatRec, Zero, Succ


def test_whnf_unfolds_natrec_on_successor() -> None:
    P = Lam(Univ(), Univ())
    z = Zero()
    s = Lam(Univ(), Lam(Univ(), Succ(Var(0))))
    term = NatRec(P=P, base=z, step=s, n=Succ(Zero()))
    result = whnf(term)

    assert result == Succ(
        NatRec(
            P=Lam(arg_ty=Univ(0), body=Univ(0)),
            base=Zero(),
            step=Lam(arg_ty=Univ(0), body=Lam(arg_ty=Univ(0), body=Succ(Var(0)))),
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


def test_whnf_stops_on_irreducible_function() -> None:
    a = Var(0)
    b = Zero()
    term = App(a, b)
    a1 = Var(0)
    b1 = Zero()
    assert whnf(term) == App(a1, b1)
