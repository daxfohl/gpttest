import pytest

import mltt.inductive.list as listm
from mltt.core.ast import Lam, Pi, Univ, Var, Term
from mltt.core.inductive_utils import nested_lam, nested_pi, apply_term
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_check
from mltt.inductive.list import ConsCtor, NilCtor
from mltt.inductive.nat import NatType, Succ, Zero


def test_infer_list_type_constructor() -> None:
    assert infer_type(listm.List) == Pi(Univ(0), Univ(0))


def test_list_nil_and_cons_type_check() -> None:
    elem_ty = NatType()
    nil_nat = listm.Nil(elem_ty)
    type_check(nil_nat, listm.ListType(elem_ty))

    cons_nat = listm.Cons(elem_ty, Zero(), nil_nat)
    type_check(cons_nat, listm.ListType(elem_ty))


def test_listrec_length_of_singleton() -> None:
    elem_ty = NatType()
    list_ty = listm.ListType(elem_ty)
    xs = listm.Cons(elem_ty, Zero(), listm.Nil(elem_ty))
    P = Lam(list_ty, NatType())
    base = Zero()
    step = nested_lam(
        elem_ty,
        list_ty,
        apply_term(P, Var(0)),
        body=Succ(Var(0)),
    )

    length_term = listm.ListRec(P, base, step, xs)

    assert normalize(length_term) == Succ(Zero())
    type_check(length_term, NatType())


@pytest.mark.parametrize(
    "elem", (Zero(), Succ(Zero()), listm.Nil(NatType()), NatType(), Univ(0), Univ(55))
)
@pytest.mark.parametrize("n", range(5))
def test_infer_type(elem: Term, n: int) -> None:
    elem_ty = infer_type(elem)
    l: Term = listm.Nil(elem_ty)
    for j in range(n):
        l = listm.Cons(elem_ty, elem, l)
    t = infer_type(l)
    assert t == listm.ListType(elem_ty)


def test_ctor_type() -> None:
    t = infer_type(NilCtor)
    # Pi x : Type. List x
    assert t == Pi(Univ(0), listm.ListType(Var(0)))
    t = infer_type(ConsCtor)
    # Pi x : Type. x -> List x -> List x
    assert t == nested_pi(
        Univ(0),
        Var(0),
        listm.ListType(Var(1)),
        return_ty=listm.ListType(Var(2)),
    )
