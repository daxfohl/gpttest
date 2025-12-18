import pytest

import mltt.inductive.fin as fin
from mltt.core.ast import Pi, Univ, Var
from mltt.core.util import apply_term, nested_pi, nested_lam
from mltt.inductive.fin import (
    FZCtor,
    FSCtor,
    fin_modulus,
    fin_to_nat,
)
from mltt.inductive.nat import NatType, Succ, Zero, numeral


def test_infer_fin_type() -> None:
    assert fin.Fin.infer_type() == Pi(NatType(), Univ(0))


def test_fz_and_fs_types() -> None:
    n0 = Zero()
    n1 = Succ(n0)
    n2 = Succ(n1)

    fz = fin.FZ(n0)
    fz.type_check(fin.FinType(n1))

    fs = fin.FS(n1, fz)
    fs.type_check(fin.FinType(n2))


def test_fin_rec_respects_index() -> None:
    # Motive specialized to the index produced by FZ 0 (i.e., Fin (Succ 0)).
    P = nested_lam(NatType(), fin.FinType(Var(0)), body=NatType())
    base = nested_lam(NatType(), body=Zero())
    step = nested_lam(
        NatType(),
        fin.FinType(Var(0)),
        apply_term(P, Var(1), Var(0)),
        body=Var(0),
    )
    k = fin.FZ(Zero())
    rec = fin.FinRec(P, base, step, k)
    assert rec.normalize() == Zero()
    rec.type_check(NatType())


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("i", range(5))
def test_infer_type(n: int, i: int) -> None:
    t = fin.of_int(i % n, n).infer_type()
    assert t == fin.FinType(numeral(n))


def test_ctor_type() -> None:
    t = FZCtor.infer_type()
    # Pi x : Nat. Fin (Succ x)
    assert t == nested_pi(NatType(), return_ty=fin.FinType(Succ(Var(0))))
    t = FSCtor.infer_type()
    # Pi x : Nat. Fin x -> Fin (Succ x)
    assert t == nested_pi(
        NatType(),
        fin.FinType(Var(0)),
        return_ty=fin.FinType(Succ(Var(1))),
    )


def test_fin_modulus() -> None:
    n = 4
    for i in range(n):
        term = fin_modulus(numeral(n), fin.of_int(i, n))
        assert term.normalize() == numeral(n)
        term.type_check(NatType())


def test_fin_to_nat() -> None:
    n = 5
    for i in range(n):
        term = fin_to_nat(numeral(n), fin.of_int(i, n))
        assert term.normalize() == numeral(i)
        term.type_check(NatType())
