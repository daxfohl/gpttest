import mltt.fin as fin
from mltt.ast import Pi, Univ
from mltt.nat import NatType, Succ, Zero
from mltt.typing import infer_type, type_check


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
