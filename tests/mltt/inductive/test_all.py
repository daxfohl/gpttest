from mltt.core.ast import Lam, Pi, Univ, Var
from mltt.core.debruijn import mk_app, mk_pis
from mltt.inductive.all import (
    AllCons,
    AllConsCtorAt,
    AllNil,
    AllNilCtorAt,
    AllType,
)
from mltt.inductive.list import Cons, ConsCtorAt, ListAt, Nil, NilCtorAt
from mltt.inductive.nat import NatType, Zero


def trivial_predicate() -> Lam:
    return Lam(NatType(), NatType())


def test_all_nil_typechecks() -> None:
    A = NatType()
    P = trivial_predicate()
    proof = AllNil(A, P)
    expected = AllType(A, P, Nil(A))
    proof.type_check(expected)
    assert proof.infer_type().type_equal(expected)


def test_all_cons_typechecks() -> None:
    A = NatType()
    P = trivial_predicate()
    xs = Nil(A)
    x = Zero()
    px = Zero()
    ih = AllNil(A, P)
    proof = AllCons(A, P, xs, x, px, ih)
    expected_list = Cons(A, x, xs)
    expected = AllType(A, P, expected_list)
    proof.type_check(expected)
    assert proof.infer_type().type_equal(expected)


def test_all_ctor_types() -> None:
    expected_nil = mk_pis(
        Univ(0),
        Pi(Var(0), Univ(0)),
        return_ty=AllType(Var(1), Var(0), mk_app(NilCtorAt(0), Var(1)), level=0),
    )
    assert AllNilCtorAt(0).infer_type().type_equal(expected_nil)

    expected_cons = mk_pis(
        Univ(0),
        Pi(Var(0), Univ(0)),
        mk_app(ListAt(0), Var(1)),
        Var(2),
        mk_app(Var(2), Var(0)),
        AllType(Var(4), Var(3), Var(2), level=0),
        return_ty=AllType(
            Var(5), Var(4), mk_app(ConsCtorAt(0), Var(5), Var(2), Var(3)), level=0
        ),
    )
    assert AllConsCtorAt(0).infer_type().type_equal(expected_cons)
