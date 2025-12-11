import pytest

import mltt.inductive.fin as fin
from mltt.core.ast import Pi, Univ, Var
from mltt.core.inductive_utils import nested_lam, nested_pi
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_check
from mltt.inductive.fin import FZCtor, FSCtor
from mltt.inductive.nat import NatType, Succ, Zero, numeral


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


def test_fin_rec_respects_index() -> None:
    # Motive specialized to the index produced by FZ 0 (i.e., Fin (Succ 0)).
    P = nested_lam(NatType(), fin.FinType(Var(0)), body=NatType())
    base = nested_lam(NatType(), body=Zero())
    step = nested_lam(NatType(), fin.FinType(Var(0)), NatType(), body=Var(0))
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
    assert t == nested_pi(NatType(), return_ty=fin.FinType(Succ(Var(0))))
    t = infer_type(FSCtor)
    # Pi x : Nat. Fin x -> Fin (Succ x)
    assert t == nested_pi(
        NatType(),
        fin.FinType(Var(0)),
        return_ty=fin.FinType(Succ(Var(1))),
    )
