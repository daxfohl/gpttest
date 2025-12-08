import pytest

import mltt.inductive.fin as fin
from mltt.core.ast import Pi, Univ, Var, Lam
from mltt.core.reduce import normalize, whnf
from mltt.inductive.fin import FZCtor, FSCtor
from mltt.inductive.nat import NatType, Succ, Zero, numeral
from mltt.core.typing import infer_type, type_check


def test_infer_fin_type() -> None:
    assert infer_type(fin.Fin) == Pi(NatType(), Univ(0))


def test_fz_and_fs_types() -> None:
    n0 = Zero()
    n1 = Succ(n0)
    n2 = Succ(n1)

    fz = fin.FZ(n0)
    assert type_check(fz, fin.FinType(n1))

    fs = fin.FS(n1, fz)
    assert type_check(fs, fin.FinType(n2))


def test_fin_rec_on_fz_reduces_to_base() -> None:
    P = Lam(fin.FinType(Zero()), NatType())
    base = Zero()
    step = Lam(fin.FinType(Zero()), Lam(NatType(), Var(0)))

    term = fin.FinRec(P, base, step, fin.FZ(Zero()))
    assert whnf(term) == base


def test_fin_rec_respects_index() -> None:
    # Motive specialized to the index produced by FZ 0 (i.e., Fin (Succ 0)).
    f1 = fin.FinType(Succ(Zero()))
    P = Lam(NatType(), Lam(fin.FinType(Succ(Var(0))), NatType()))
    base = Zero()
    step = Lam(f1, Lam(NatType(), Var(0)))
    k = fin.FZ(Zero())
    rec = fin.FinRec(P, base, step, k)
    assert normalize(rec) == Zero()
    assert type_check(rec, NatType())


@pytest.mark.parametrize("n", range(1, 5))
@pytest.mark.parametrize("i", range(5))
def test_infer_type(n: int, i: int) -> None:
    t = infer_type(fin.of_int(i, n))
    assert t == fin.FinType(numeral(n))


def test_ctor_type() -> None:
    t = infer_type(FZCtor)
    # Pi x : Nat. Fin (Succ x)
    assert t == Pi(NatType(), fin.FinType(Succ(Var(0))))
    t = infer_type(FSCtor)
    # Pi x : Nat. Fin x -> Fin (Succ x)
    assert t == Pi(NatType(), Pi(fin.FinType(Var(0)), fin.FinType(Succ(Var(1)))))
