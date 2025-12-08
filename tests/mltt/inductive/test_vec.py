import pytest

import mltt.inductive.vec as vec
from mltt.core.ast import Pi, Univ, Term, Var, Lam, App
from mltt.core.reduce import normalize, whnf
from mltt.core.typing import infer_type, type_check
from mltt.inductive.nat import NatType, Succ, Zero, numeral, add_terms
from mltt.inductive.vec import VecType


def test_infer_vec_type() -> None:
    assert infer_type(vec.Vec) == Pi(Univ(0), Pi(NatType(), Univ(0)))


def test_nil_has_zero_length() -> None:
    elem_ty = NatType()
    nil = vec.Nil(elem_ty)
    assert type_check(nil, vec.VecType(elem_ty, Zero()))


def test_cons_increments_length() -> None:
    elem_ty = NatType()
    tail = vec.Nil(elem_ty)
    cons = vec.Cons(elem_ty, Zero(), Zero(), tail)
    assert type_check(cons, vec.VecType(elem_ty, Succ(Zero())))


def test_vec_rec_on_nil_reduces_to_base() -> None:
    elem_ty = NatType()
    P = Lam(vec.VecType(elem_ty, Zero()), NatType())
    base = Zero()
    step = Lam(elem_ty, Lam(vec.VecType(elem_ty, Zero()), Lam(NatType(), Var(0))))

    term = vec.VecRec(P, base, step, vec.Nil(elem_ty))
    assert whnf(term) == base


@pytest.mark.parametrize("vec_len", range(4))
@pytest.mark.parametrize("b", range(4))
@pytest.mark.parametrize("v", range(4))
def test_vec_rec_preserves_length_index1(vec_len: int, b: int, v: int) -> None:
    elem_ty = NatType()
    P = Lam(
        NatType(),  # n : Nat
        Lam(
            vec.VecType(elem_ty, Var(0)),
            NatType(),
        ),  # xs : Vec A n
    )

    base = numeral(b)
    step = Lam(
        elem_ty,  # x : A
        Lam(
            vec.VecType(elem_ty, Var(1)),  # xs : Vec A n (Var(1) = n)
            Lam(
                App(Var(0), P),  # ih : P xs
                add_terms(Var(0), Var(2)),  # ih + x
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
    P = Lam(
        Univ(0), Lam(NatType(), Lam(vec.VecType(Var(1), Succ(Succ(Zero()))), NatType()))
    )

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


@pytest.mark.parametrize(
    "elem", (Zero(), Succ(Zero()), vec.Nil(NatType()), NatType(), Univ(0), Univ(55))
)
@pytest.mark.parametrize("n", range(5))
def test_infer_type(elem: Term, n: int) -> None:
    elem_ty = infer_type(elem)
    a = vec.Nil(elem_ty)
    for i in range(n):
        a = vec.Cons(elem_ty, numeral(i), elem, a)
    t = infer_type(a)
    assert t == VecType(elem_ty, numeral(n))


def test_ctor_type() -> None:
    t = infer_type(vec.NilCtor)
    # Pi x : Type. Nat -> Vec x Zero
    assert t == Pi(Univ(0), Pi(NatType(), vec.VecType(Var(1), Zero())))
    t = infer_type(vec.ConsCtor)
    # Pi x : Type. Pi x1 : Nat. x -> Vec x x1 -> Vec x (Succ x1)
    assert t == Pi(
        Univ(0),
        Pi(
            NatType(),
            Pi(
                Var(1),
                Pi(vec.VecType(Var(2), Var(1)), vec.VecType(Var(3), Succ(Var(2)))),
            ),
        ),
    )
