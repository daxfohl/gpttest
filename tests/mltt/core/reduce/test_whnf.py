from mltt.core.ast import App, Var
from mltt.core.reduce.whnf import whnf
from mltt.core.util import nested_lam
from mltt.inductive.eq import IdElim, Refl
from mltt.inductive.nat import NatRec, NatType, Zero, Succ


def test_whnf_unfolds_natrec_on_successor() -> None:
    A = NatType()
    z = Zero()
    s = nested_lam(NatType(), NatType(), body=Succ(Var(0)))
    term = NatRec(A=A, base=z, step=s, n=Succ(Zero()))
    result = whnf(term)

    assert result == Succ(
        NatRec(
            A=NatType(),
            base=Zero(),
            step=nested_lam(NatType(), NatType(), body=Succ(Var(0))),
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
    term = App(Var(0), Zero())
    assert whnf(term) == App(Var(0), Zero())
