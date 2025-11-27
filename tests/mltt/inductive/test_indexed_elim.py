import mltt.inductive.fin as fin
import mltt.inductive.vec as vec
from mltt.core.ast import Lam, Var
from mltt.core.reduce import normalize, whnf
from mltt.core.typing import type_check
from mltt.inductive.nat import NatType, Zero, Succ


def test_vec_rec_on_nil_reduces_to_base() -> None:
    elem_ty = NatType()
    P = Lam(vec.VecType(elem_ty, Zero()), NatType())
    base = Zero()
    step = Lam(elem_ty, Lam(vec.VecType(elem_ty, Zero()), Lam(NatType(), Var(0))))

    term = vec.VecRec(elem_ty, P, base, step, vec.Nil(elem_ty))
    assert whnf(term) == base


def test_fin_rec_on_fz_reduces_to_base() -> None:
    P = Lam(fin.FinType(Zero()), NatType())
    base = Zero()
    step = Lam(fin.FinType(Zero()), Lam(NatType(), Var(0)))

    term = fin.FinRec(P, base, step, fin.FZ(Zero()))
    assert whnf(term) == base


def test_vec_rec_preserves_length_index() -> None:
    elem_ty = NatType()
    # Motive specialized to length 0 so it matches Nil's result index.
    P = Lam(vec.VecType(elem_ty, Zero()), NatType())

    base = Var(0)  # P (Nil A) = Nat
    step = Lam(
        elem_ty,
        Lam(
            vec.VecType(elem_ty, Zero()),
            Lam(NatType(), Var(0)),  # ignore IH; return Nat
        ),
    )

    xs = vec.Nil(elem_ty)
    rec = vec.VecRec(elem_ty, P, base, step, xs)
    assert type_check(rec, NatType())
    assert normalize(rec) == Zero()


def test_fin_rec_respects_index() -> None:
    # Motive specialized to the index produced by FZ 0 (i.e., Fin (Succ 0)).
    P = Lam(fin.FinType(Succ(Zero())), NatType())
    base = Zero()
    step = Lam(fin.FinType(Zero()), Lam(NatType(), Var(0)))

    k = fin.FZ(Zero())
    rec = fin.FinRec(P, base, step, k)
    assert type_check(rec, NatType())
    assert normalize(rec) == Zero()
