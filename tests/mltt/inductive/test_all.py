from mltt.inductive.all import (
    AllAt,
    AllCons,
    AllConsCtor,
    AllNil,
    AllNilCtor,
    AllType,
)
from mltt.inductive.list import Cons, ConsCtor, List, ListAt, Nil, NilCtor
from mltt.inductive.nat import NatType, Zero
from mltt.kernel.ast import Lam, Pi, Univ, Var
from mltt.kernel.tel import mk_app, mk_pis


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
        return_ty=AllType(Var(1), Var(0), mk_app(NilCtor, Var(1))),
    )
    assert AllNilCtor.infer_type().type_equal(expected_nil)

    expected_cons = mk_pis(
        Univ(0),
        Pi(Var(0), Univ(0)),
        mk_app(List, Var(1)),
        Var(2),
        mk_app(Var(2), Var(0)),
        AllType(Var(4), Var(3), Var(2)),
        return_ty=AllType(Var(5), Var(4), mk_app(ConsCtor, Var(5), Var(2), Var(3))),
    )
    assert AllConsCtor.infer_type().type_equal(expected_cons)


def test_infer_all_type_constructor_at_level() -> None:
    expected = mk_pis(
        Univ(1),
        Pi(Var(0), Univ(1)),
        mk_app(ListAt(1), Var(1)),
        return_ty=Univ(1),
    )
    assert AllAt(1).infer_type().type_equal(expected)
