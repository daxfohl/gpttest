import pytest

import mltt.inductive.list as listm
from mltt.core.ast import Lam, Pi, Univ, Var, Term
from mltt.core.debruijn import mk_app, mk_pis, mk_lams
from mltt.inductive.list import ConsCtor, NilCtor
from mltt.inductive.nat import NatType, Succ, Zero


def test_infer_list_type_constructor() -> None:
    assert listm.List.infer_type() == Pi(Univ(0), Univ(0))


def test_list_nil_and_cons_type_check() -> None:
    elem_ty = NatType()
    nil_nat = listm.Nil(elem_ty)
    nil_nat.type_check(listm.ListType(elem_ty))

    cons_nat = listm.Cons(elem_ty, Zero(), nil_nat)
    cons_nat.type_check(listm.ListType(elem_ty))


def test_listrec_length_of_singleton() -> None:
    elem_ty = NatType()
    list_ty = listm.ListType(elem_ty)
    xs = listm.Cons(elem_ty, Zero(), listm.Nil(elem_ty))
    P = Lam(list_ty, NatType())
    base = Zero()
    step = mk_lams(
        elem_ty,
        list_ty,
        mk_app(P, Var(0)),
        body=Succ(Var(0)),
    )

    length_term = listm.ListElim(P, base, step, xs)

    assert length_term.normalize() == Succ(Zero())
    length_term.type_check(NatType())


@pytest.mark.parametrize(
    "elem", (Zero(), Succ(Zero()), listm.Nil(NatType()), NatType(), Univ(0), Univ(55))
)
@pytest.mark.parametrize("n", range(5))
def test_infer_type(elem: Term, n: int) -> None:
    elem_ty = elem.infer_type()
    l: Term = listm.Nil(elem_ty)
    for j in range(n):
        l = listm.Cons(elem_ty, elem, l)
    t = l.infer_type()
    assert t == listm.ListType(elem_ty)


def test_ctor_type() -> None:
    t = NilCtor.infer_type()
    # Pi x : Type. List x
    assert t == Pi(Univ(0), listm.ListType(Var(0)))
    t = ConsCtor.infer_type()
    # Pi x : Type. x -> List x -> List x
    assert t == mk_pis(
        Univ(0),
        Var(0),
        listm.ListType(Var(1)),
        return_ty=listm.ListType(Var(2)),
    )
