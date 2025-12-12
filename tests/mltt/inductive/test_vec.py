import pytest

import mltt.inductive.vec as vec
from mltt.core.ast import Lam, Term, Univ, Var
from mltt.core.inductive_utils import nested_lam, nested_pi, apply_term
from mltt.core.reduce import normalize, whnf
from mltt.core.typing import infer_type, type_check
from mltt.inductive.nat import NatType, Succ, Zero, numeral, add_terms
from mltt.inductive.vec import VecType


def test_infer_vec_type() -> None:
    assert infer_type(vec.Vec) == nested_pi(Univ(0), NatType(), return_ty=Univ(0))


def test_nil_has_zero_length() -> None:
    elem_ty = NatType()
    nil = vec.Nil(elem_ty)
    assert type_check(nil, vec.VecType(elem_ty, Zero()))


def test_cons_increments_length() -> None:
    elem_ty = NatType()
    tail = vec.Nil(elem_ty)
    cons = vec.Cons(elem_ty, Zero(), Zero(), tail)
    assert type_check(cons, vec.VecType(elem_ty, Succ(Zero())))


def test_vec_rec_on_nil_reduces_to_zero() -> None:
    elem_ty = NatType()
    P = nested_lam(NatType(), vec.VecType(elem_ty, Var(0)), body=NatType())
    base = Zero()
    step = nested_lam(elem_ty, vec.VecType(elem_ty, Zero()), NatType(), body=Var(0))

    term = vec.VecRec(P, base, step, vec.Nil(elem_ty))
    assert whnf(term) == Zero()


@pytest.mark.parametrize("vec_len", range(4))
@pytest.mark.parametrize("b", range(4))
@pytest.mark.parametrize("v", range(4))
def test_vec_rec_preserves_length_index1(vec_len: int, b: int, v: int) -> None:
    elem_ty = NatType()
    P = nested_lam(Univ(0), NatType(), vec.VecType(elem_ty, Var(0)), body=NatType())

    base = numeral(b)
    step = nested_lam(
        elem_ty,  # x : A
        vec.VecType(elem_ty, Var(1)),  # xs : Vec A n (Var(1) = n)
        apply_term(P, Var(2), Var(0)),  # ih : P xs
        body=add_terms(Var(0), Var(2)),  # ih + x
    )

    xs: Term = vec.Nil(elem_ty)
    for i in range(vec_len):
        xs = vec.Cons(elem_ty, numeral(i), numeral(v), xs)
    rec = vec.VecRec(P, base, step, xs)
    normalized = normalize(rec)
    assert normalized == numeral(v * vec_len + b)
    assert type_check(rec, NatType())


def test_vec_rec_preserves_length_index() -> None:
    elem_ty = NatType()
    # Motive specialized to length 0 so it matches Nil's result index.
    P = nested_lam(
        Univ(0),
        NatType(),
        vec.VecType(Var(1), Var(0)),
        body=NatType(),
    )

    base = Succ(Zero())  # P (Nil A) = Nat
    step = nested_lam(
        elem_ty,
        vec.VecType(elem_ty, Var(1)),
        NatType(),
        body=Var(2),  # ignore IH; return Nat
    )

    xs: Term = vec.Nil(elem_ty)
    # xs = vec.Cons(elem_ty, Zero(), Zero(), xs)  # say Vec A 1
    # xs = vec.Cons(elem_ty, Succ(Zero()), Succ(Zero()), xs)  # say Vec A 1

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
    vector = vec.Nil(elem_ty)
    for i in range(n):
        vector = vec.Cons(elem_ty, numeral(i), elem, vector)
    t = infer_type(vector)
    assert t == VecType(elem_ty, numeral(n))


def test_ctor_type() -> None:
    t = infer_type(vec.NilCtor)
    # Pi x : Type. Nat -> Vec x Zero
    assert t == nested_pi(Univ(0), NatType(), return_ty=vec.VecType(Var(1), Zero()))
    t = infer_type(vec.ConsCtor)
    # Pi x : Type. Pi x1 : Nat. x -> Vec x x1 -> Vec x (Succ x1)
    assert t == nested_pi(
        Univ(0),
        NatType(),
        Var(1),
        vec.VecType(Var(2), Var(1)),
        return_ty=vec.VecType(Var(3), Succ(Var(2))),
    )
