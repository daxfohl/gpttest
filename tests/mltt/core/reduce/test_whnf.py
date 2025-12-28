from mltt.kernel.ast import App, Var
from mltt.kernel.debruijn import mk_lams
from mltt.inductive.eq import IdElim, Refl
from mltt.inductive.nat import NatRec, NatType, Zero, Succ


def test_whnf_unfolds_natrec_on_successor() -> None:
    A = NatType()
    z = Zero()
    s = mk_lams(NatType(), NatType(), body=Succ(Var(0)))
    term = NatRec(A=A, base=z, step=s, n=Succ(Zero()))
    result = term.whnf()

    assert result == Succ(
        NatRec(
            A=NatType(),
            base=Zero(),
            step=mk_lams(NatType(), NatType(), body=Succ(Var(0))),
            n=Zero(),
        )
    )


def test_whnf_simplifies_identity_elimination_on_refl() -> None:
    term = IdElim(
        P=Var(2),
        d=Var(3),
        p=Refl(Var(5), Var(6)),
    )
    assert term.whnf() == Var(3)


def test_whnf_stops_on_irreducible_function() -> None:
    term = App(Var(0), Zero())
    assert term.whnf() == App(Var(0), Zero())
