import pytest

import mltt.inductive.fin as fin
import mltt.inductive.vec as vec
from mltt.core.ast import Lam, Var, Term
from mltt.core.reduce import normalize, whnf
from mltt.core.typing import type_check
from mltt.inductive.nat import (
    NatType,
    Zero,
    Succ,
    numeral,
    add_terms,
)


def test_vec_rec_on_nil_reduces_to_base() -> None:
    elem_ty = NatType()
    P = Lam(vec.VecType(elem_ty, Zero()), NatType())
    base = Zero()
    step = Lam(elem_ty, Lam(vec.VecType(elem_ty, Zero()), Lam(NatType(), Var(0))))

    term = vec.VecRec(P, base, step, vec.Nil(elem_ty))
    assert whnf(term) == base


def test_fin_rec_on_fz_reduces_to_base() -> None:
    P = Lam(fin.FinType(Zero()), NatType())
    base = Zero()
    step = Lam(fin.FinType(Zero()), Lam(NatType(), Var(0)))

    term = fin.FinRec(P, base, step, fin.FZ(Zero()))
    assert whnf(term) == base


@pytest.mark.parametrize("vec_len", range(4))
@pytest.mark.parametrize("b", range(4))
@pytest.mark.parametrize("v", range(4))
def test_vec_rec_preserves_length_index1(vec_len: int, b: int, v: int) -> None:
    elem_ty = NatType()
    P = Lam(
        NatType(),  # n : Nat
        Lam(vec.VecType(elem_ty, Var(0)), NatType()),  # xs : Vec A n
    )

    base = numeral(b)
    step = Lam(
        NatType(),  # n : Nat
        Lam(
            elem_ty,  # x : A
            Lam(
                vec.VecType(elem_ty, Var(1)),  # xs : Vec A n (Var(1) = n)
                Lam(
                    NatType(),  # acc : Nat
                    add_terms(Var(0), Var(2)),  # acc + x
                ),
            ),
        ),
    )

    xs: Term = vec.Nil(elem_ty)
    for i in range(vec_len):
        xs = vec.Cons(elem_ty, numeral(i), numeral(v), xs)
    rec = vec.VecRec(P, base, step, xs)
    normalized = normalize(rec)
    assert normalized == numeral(v * vec_len + b)
    # assert type_check(rec, NatType())


def test_vec_rec_preserves_length_index() -> None:
    elem_ty = NatType()
    # Motive specialized to length 0 so it matches Nil's result index.
    P = Lam(vec.VecType(elem_ty, Succ(Succ(Zero()))), NatType())

    base = Succ(Zero())  # P (Nil A) = Nat
    step = Lam(
        elem_ty,
        Lam(
            vec.VecType(elem_ty, Succ(Succ(Zero()))),
            Lam(NatType(), Var(2)),  # ignore IH; return Nat
        ),
    )

    xs: Term = vec.Nil(elem_ty)
    xs = vec.Cons(elem_ty, Zero(), Zero(), xs)  # say Vec A 1
    xs = vec.Cons(elem_ty, Succ(Zero()), Succ(Zero()), xs)  # say Vec A 1

    rec = vec.VecRec(P, base, step, xs)
    normalized = normalize(rec)
    assert normalized == Succ(Zero())
    assert type_check(rec, NatType())


def test_fin_rec_respects_index() -> None:
    # a = _ctor_type(ZeroCtor)
    # b = _ctor_type(SuccCtor).ty
    # print(a==b)
    # print('hi')
    # return
    # Motive specialized to the index produced by FZ 0 (i.e., Fin (Succ 0)).
    print()
    P = Lam(fin.FinType(Succ(Zero())), NatType())
    base = Zero()
    step = Lam(fin.FinType(Zero()), Lam(NatType(), Var(0)))
    k = fin.FZ(Zero())
    rec = fin.FinRec(P, base, step, k)
    assert normalize(rec) == Zero()
    print("asssert")
    assert type_check(rec, NatType())
