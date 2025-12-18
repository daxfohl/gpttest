from mltt.core.ast import Lam, Pi, Univ, Var
from mltt.core.util import apply_term, nested_pi
from mltt.inductive.all import AllCons, AllConsCtor, AllNil, AllNilCtor, AllType
from mltt.inductive.list import Cons, ConsCtor, List, Nil, NilCtor
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
    expected_nil = nested_pi(
        Univ(0),
        Pi(Var(0), Univ(0)),
        return_ty=AllType(Var(1), Var(0), apply_term(NilCtor, Var(1))),
    )
    assert AllNilCtor.infer_type().type_equal(expected_nil)

    expected_cons = nested_pi(
        Univ(0),
        Pi(Var(0), Univ(0)),
        apply_term(List, Var(1)),
        Var(2),
        apply_term(Var(2), Var(0)),
        AllType(Var(4), Var(3), Var(2)),
        return_ty=AllType(Var(5), Var(4), apply_term(ConsCtor, Var(5), Var(2), Var(3))),
    )
    assert AllConsCtor.infer_type().type_equal(expected_cons)
