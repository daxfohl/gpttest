import pytest

import mltt.inductive.list as listm
from mltt.core.ast import App, Lam, Pi, Univ, Var, Term
from mltt.core.reduce import normalize
from mltt.core.typing import infer_type, type_check
from mltt.inductive import nat
from mltt.inductive.list import ConsCtor, NilCtor
from mltt.inductive.nat import NatType, Succ, Zero, numeral


def test_infer_list_type_constructor() -> None:
    assert infer_type(listm.List) == Pi(Univ(0), Univ(0))


def test_list_nil_and_cons_type_check() -> None:
    elem_ty = NatType()
    nil_nat = listm.Nil(elem_ty)
    assert type_check(nil_nat, listm.ListType(elem_ty))

    cons_nat = listm.Cons(elem_ty, Zero(), nil_nat)
    assert type_check(cons_nat, listm.ListType(elem_ty))


def test_listrec_length_of_singleton() -> None:
    elem_ty = NatType()
    list_ty = listm.ListType(elem_ty)
    xs = listm.Cons(elem_ty, Zero(), listm.Nil(elem_ty))
    P = Lam(Univ(0), Lam(listm.ListType(Var(0)), NatType()))
    base = Zero()
    step = Lam(
        elem_ty,
        Lam(
            list_ty,
            Lam(
                App(P, Var(0)),
                Succ(Var(0)),
            ),
        ),
    )

    length_term = listm.ListRec(P, base, step, xs)

    assert normalize(length_term) == Succ(Zero())
    assert type_check(length_term, NatType())


@pytest.mark.parametrize(
    "elem", (Zero(), Succ(Zero()), listm.Nil(NatType()), NatType(), Univ(0), Univ(55))
)
@pytest.mark.parametrize("n", range(5))
def test_infer_type(elem: Term, n: int) -> None:
    elem_ty = infer_type(elem)
    a = listm.Nil(elem_ty)
    for j in range(n):
        a = listm.Cons(elem_ty, elem, a)
    t = infer_type(a)
    assert t == listm.ListType(elem_ty)


def test_ctor_type() -> None:
    t = infer_type(NilCtor)
    # Pi x : Type. List x
    assert t == Pi(Univ(0), listm.ListType(Var(0)))
    t = infer_type(ConsCtor)
    # Pi x : Type. x -> List x -> List x
    assert t == Pi(
        Univ(0), Pi(Var(0), Pi(listm.ListType(Var(1)), listm.ListType(Var(2))))
    )
